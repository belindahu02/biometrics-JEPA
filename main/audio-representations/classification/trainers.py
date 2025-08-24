# =============================================
# Enhanced trainers.py with Model Checkpointing
# =============================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from backbones import ResNetBlockFinal
from data_loader import load_spectrogram_data_as_1d, norma, user_data_split
import os
import json
from datetime import datetime


def spectrogram_trainer(samples_per_user, data_path, user_ids, conversion_method='pca',
                        batch_size=8, epochs=100, lr=0.001, device=None,
                        save_model_checkpoints=True, checkpoint_every=10):
    """
    Enhanced PyTorch spectrogram trainer with model checkpointing.

    Args:
        samples_per_user: Number of samples per user for training
        data_path: Path to spectrogram data
        user_ids: List of user IDs to include
        conversion_method: Method for converting spectrograms to 1D
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Initial learning rate
        device: torch device ('cuda' or 'cpu')
        save_model_checkpoints: Whether to save model checkpoints during training
        checkpoint_every: Save checkpoint every N epochs
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    frame_size = 520
    target_features = 24

    # Create checkpoint directory
    if save_model_checkpoints:
        checkpoint_dir = os.path.join(os.path.dirname(data_path), "model_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create unique identifier for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{conversion_method}_{samples_per_user}samples_{timestamp}"
        run_checkpoint_dir = os.path.join(checkpoint_dir, run_id)
        os.makedirs(run_checkpoint_dir, exist_ok=True)

        print(f"Model checkpoints will be saved to: {run_checkpoint_dir}")

    # --------------------------
    # Load and preprocess data
    # --------------------------
    print("Loading and preprocessing data...")
    x_train, y_train, x_val, y_val, x_test, y_test, sessions = load_spectrogram_data_as_1d(
        data_path, user_ids, conversion_method=conversion_method, target_features=target_features
    )

    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    x_train, y_train = user_data_split(x_train, y_train, samples_per_user=samples_per_user)

    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Move channels to PyTorch format: (batch, channels, seq_len)
    x_train = x_train.permute(0, 2, 1)
    x_val = x_val.permute(0, 2, 1)
    x_test = x_test.permute(0, 2, 1)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

    num_classes = len(np.unique(y_train.numpy()))

    # --------------------------
    # Build model
    # --------------------------
    class SpectrogramResNet(nn.Module):
        def __init__(self, input_channels, num_classes, ks=3, con=3):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 16 * con, kernel_size=ks, padding='same')
            self.bn1 = nn.BatchNorm1d(16 * con)
            self.resblock = ResNetBlockFinal(16 * con, out_channels=32 * con, kernel_size=ks)
            self.fc1 = nn.Linear(32 * con, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = self.resblock(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

    model = SpectrogramResNet(input_channels=target_features, num_classes=num_classes).to(device)
    print(model)

    # --------------------------
    # Training setup
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs_completed': 0
    }

    def save_training_checkpoint(epoch, model, optimizer, train_loss, train_acc, val_acc, is_best=False):
        """Save training checkpoint"""
        if not save_model_checkpoints:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'training_history': training_history,
            'model_config': {
                'input_channels': target_features,
                'num_classes': num_classes,
                'samples_per_user': samples_per_user,
                'conversion_method': conversion_method
            }
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(run_checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if is_best:
            best_model_path = os.path.join(run_checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved (val_acc: {val_acc:.4f})")

        # Keep only last 3 regular checkpoints to save space
        checkpoints = [f for f in os.listdir(run_checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if len(checkpoints) > 3:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for old_checkpoint in checkpoints[:-3]:
                os.remove(os.path.join(run_checkpoint_dir, old_checkpoint))

    # Check for existing checkpoint to resume from
    resume_epoch = 0
    if save_model_checkpoints:
        existing_checkpoints = [f for f in os.listdir(run_checkpoint_dir)
                                if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if existing_checkpoints:
            latest_checkpoint = max(existing_checkpoints,
                                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(run_checkpoint_dir, latest_checkpoint)
            print(f"Found existing checkpoint: {latest_checkpoint}")

            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                resume_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint['best_val_acc']
                training_history = checkpoint['training_history']
                print(f"Resuming training from epoch {resume_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")
                resume_epoch = 0

    # --------------------------
    # Training loop
    # --------------------------
    print(f"Starting training from epoch {resume_epoch}...")

    for epoch in range(resume_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        epoch_loss = running_loss / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)
        val_acc = correct_val / total_val

        # Update training history
        training_history['train_loss'].append(epoch_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['epochs_completed'] = epoch + 1

        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        # Save checkpoint
        if save_model_checkpoints and (epoch + 1) % checkpoint_every == 0:
            save_training_checkpoint(epoch, model, optimizer, epoch_loss, train_acc, val_acc, is_best)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Save final checkpoint
    if save_model_checkpoints:
        save_training_checkpoint(epochs - 1, model, optimizer, epoch_loss, train_acc, val_acc,
                                 val_acc >= best_val_acc)

        # Save training history
        history_file = os.path.join(run_checkpoint_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {
                'train_loss': [float(x) for x in training_history['train_loss']],
                'train_acc': [float(x) for x in training_history['train_acc']],
                'val_acc': [float(x) for x in training_history['val_acc']],
                'epochs_completed': int(training_history['epochs_completed']),
                'best_val_acc': float(best_val_acc)
            }
            json.dump(history_json, f, indent=2)

    # --------------------------
    # Load best model and evaluate
    # --------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluation on test set
    model.eval()
    correct_test, total_test = 0, 0
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            correct_test += (preds == yb).sum().item()
            total_test += yb.size(0)
    test_acc = correct_test / total_test
    print(f"Test Accuracy: {test_acc:.4f}")

    # Compute Cohen's Kappa
    all_preds = np.concatenate(all_preds)
    y_true = y_test.numpy()
    po = np.mean(all_preds == y_true)
    pe = np.sum(np.bincount(y_true) * np.bincount(all_preds)) / (len(y_true) ** 2)
    kappa_score = (po - pe) / (1 - pe)
    print(f"Cohen's Kappa: {kappa_score:.4f}")

    # Save final results
    if save_model_checkpoints:
        final_results = {
            'test_accuracy': float(test_acc),
            'kappa_score': float(kappa_score),
            'best_val_acc': float(best_val_acc),
            'training_completed': True,
            'total_epochs': epochs,
            'samples_per_user': samples_per_user,
            'conversion_method': conversion_method
        }
        results_file = os.path.join(run_checkpoint_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"Training completed. All files saved to: {run_checkpoint_dir}")

    return test_acc, kappa_score