# =============================================
# Enhanced trainers.py with Comprehensive Logging
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
import time
import logging
import matplotlib.pyplot as plt


def setup_logging(log_dir):
    """Setup comprehensive logging for training"""
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training Loss
    axes[0, 0].plot(history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Training vs Validation Accuracy
    axes[0, 1].plot(history['train_acc'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Learning Rate (if available)
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'], 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')

    # Validation Loss (if available)
    if 'val_loss' in history:
        axes[1, 1].plot(history['train_loss'], 'b-', label='Training Loss')
        axes[1, 1].plot(history['val_loss'], 'r-', label='Validation Loss')
        axes[1, 1].set_title('Training vs Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        # Overfitting indicator
        axes[1, 1].plot(np.array(history['train_acc']) - np.array(history['val_acc']), 'purple',
                        label='Train-Val Accuracy Gap')
        axes[1, 1].set_title('Overfitting Indicator (Train-Val Gap)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_metrics(outputs, targets):
    """Calculate additional training metrics"""
    with torch.no_grad():
        preds = outputs.argmax(dim=1)

        # Basic accuracy
        accuracy = (preds == targets).float().mean().item()

        # Per-class accuracy
        num_classes = outputs.shape[1]
        per_class_acc = []
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == targets[class_mask]).float().mean().item()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)

        # Confidence statistics
        probs = F.softmax(outputs, dim=1)
        max_probs = probs.max(dim=1)[0]
        avg_confidence = max_probs.mean().item()

        return accuracy, per_class_acc, avg_confidence


def spectrogram_trainer(samples_per_user, data_path, user_ids, conversion_method='pca',
                        batch_size=8, epochs=200, lr=0.001, device=None,
                        save_model_checkpoints=True, checkpoint_every=10):
    """
    Enhanced PyTorch spectrogram trainer with comprehensive logging.

    Args:
        samples_per_user: Number of samples per user for training
        data_path: Path to spectrogram data
        user_ids: List of user IDs to include
        conversion_method: Method for converting spectrograms to 1D
        batch_size: Batch size for training
        epochs: Number of training epochs (increased from 100 to 200)
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

        # Setup logging
        logger = setup_logging(run_checkpoint_dir)
        logger.info(f"Starting training run: {run_id}")
        logger.info(f"Model checkpoints will be saved to: {run_checkpoint_dir}")
    else:
        logger = setup_logging("./logs")

    # --------------------------
    # Load and preprocess data
    # --------------------------
    logger.info("Loading and preprocessing data...")
    start_time = time.time()

    x_train, y_train, x_val, y_val, x_test, y_test, sessions = load_spectrogram_data_as_1d(
        data_path, user_ids, conversion_method=conversion_method, target_features=target_features
    )

    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    x_train, y_train = user_data_split(x_train, y_train, samples_per_user=samples_per_user)

    data_load_time = time.time() - start_time
    logger.info(f"Data loading completed in {data_load_time:.2f}s")
    logger.info(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}")
    logger.info(f"Class distribution - Val: {np.bincount(y_val)}")
    logger.info(f"Class distribution - Test: {np.bincount(y_test)}")

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
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Batch size: {batch_size}, Total batches per epoch: {len(train_loader)}")

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

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: {model}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --------------------------
    # Training setup
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6
    )

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    early_stopping_patience = 25

    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'per_class_train_acc': [],
        'per_class_val_acc': [],
        'train_confidence': [],
        'val_confidence': [],
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
            'scheduler_state_dict': scheduler.state_dict(),
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
            logger.info(f"New best model saved (val_acc: {val_acc:.4f})")

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
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")

            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                resume_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint['best_val_acc']
                training_history = checkpoint['training_history']
                logger.info(f"Resuming training from epoch {resume_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")
                resume_epoch = 0

    # --------------------------
    # Training loop
    # --------------------------
    logger.info(f"Starting training from epoch {resume_epoch} to {epochs}...")
    logger.info(f"Early stopping patience: {early_stopping_patience}")

    training_start_time = time.time()

    for epoch in range(resume_epoch, epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        all_train_outputs = []
        all_train_targets = []

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            all_train_outputs.append(outputs.detach())
            all_train_targets.append(yb.detach())

        # Calculate training metrics
        all_train_outputs = torch.cat(all_train_outputs)
        all_train_targets = torch.cat(all_train_targets)
        train_acc, per_class_train_acc, train_confidence = calculate_metrics(all_train_outputs, all_train_targets)
        epoch_loss = running_loss / len(x_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_outputs = []
        all_val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                all_val_outputs.append(outputs)
                all_val_targets.append(yb)

        all_val_outputs = torch.cat(all_val_outputs)
        all_val_targets = torch.cat(all_val_targets)
        val_acc, per_class_val_acc, val_confidence = calculate_metrics(all_val_outputs, all_val_targets)
        val_loss /= len(x_val)

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        # Update training history
        training_history['train_loss'].append(epoch_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(current_lr)
        training_history['per_class_train_acc'].append(per_class_train_acc)
        training_history['per_class_val_acc'].append(per_class_val_acc)
        training_history['train_confidence'].append(train_confidence)
        training_history['val_confidence'].append(val_confidence)
        training_history['epochs_completed'] = epoch + 1

        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Detailed logging every epoch
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            logger.info(f"Epoch {epoch + 1:3d}/{epochs} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"LR: {current_lr:.2e} | "
                        f"Train Loss: {epoch_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f} | "
                        f"Train Conf: {train_confidence:.3f} | "
                        f"Val Conf: {val_confidence:.3f} | "
                        f"Best Val: {best_val_acc:.4f}" +
                        (" 🌟" if is_best else ""))

            # Per-class accuracy logging (less frequent to avoid spam)
            if (epoch + 1) % 20 == 0:
                logger.info(f"Per-class Train Acc: {[f'{acc:.3f}' for acc in per_class_train_acc]}")
                logger.info(f"Per-class Val Acc:   {[f'{acc:.3f}' for acc in per_class_val_acc]}")

        # Save checkpoint
        if save_model_checkpoints and (epoch + 1) % checkpoint_every == 0:
            save_training_checkpoint(epoch, model, optimizer, epoch_loss, train_acc, val_acc, is_best)

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

        # Plot training curves every 25 epochs
        if save_model_checkpoints and (epoch + 1) % 25 == 0:
            plot_path = os.path.join(run_checkpoint_dir, f'training_curves_epoch_{epoch + 1}.png')
            plot_training_curves(training_history, plot_path)

    # Training completed
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time:.2f}s")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save final checkpoint
    if save_model_checkpoints:
        save_training_checkpoint(epoch, model, optimizer, epoch_loss, train_acc, val_acc,
                                 val_acc >= best_val_acc)

        # Save training history
        history_file = os.path.join(run_checkpoint_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {
                'train_loss': [float(x) for x in training_history['train_loss']],
                'train_acc': [float(x) for x in training_history['train_acc']],
                'val_loss': [float(x) for x in training_history['val_loss']],
                'val_acc': [float(x) for x in training_history['val_acc']],
                'learning_rate': [float(x) for x in training_history['learning_rate']],
                'epochs_completed': int(training_history['epochs_completed']),
                'best_val_acc': float(best_val_acc),
                'total_training_time': float(total_training_time),
                'early_stopped': epochs_without_improvement >= early_stopping_patience
            }
            json.dump(history_json, f, indent=2)

        # Create final training curves plot
        final_plot_path = os.path.join(run_checkpoint_dir, 'final_training_curves.png')
        plot_training_curves(training_history, final_plot_path)

    # --------------------------
    # Load best model and evaluate
    # --------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model for final evaluation")

    # Evaluation on test set
    logger.info("Starting final evaluation on test set...")
    model.eval()
    test_loss = 0.0
    all_test_outputs = []
    all_test_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item() * xb.size(0)
            all_test_outputs.append(outputs)
            all_test_targets.append(yb)

    all_test_outputs = torch.cat(all_test_outputs)
    all_test_targets = torch.cat(all_test_targets)
    test_acc, per_class_test_acc, test_confidence = calculate_metrics(all_test_outputs, all_test_targets)
    test_loss /= len(x_test)

    # Compute Cohen's Kappa
    all_preds = all_test_outputs.argmax(dim=1).cpu().numpy()
    y_true = all_test_targets.cpu().numpy()
    po = np.mean(all_preds == y_true)
    pe = np.sum(np.bincount(y_true) * np.bincount(all_preds)) / (len(y_true) ** 2)
    kappa_score = (po - pe) / (1 - pe) if pe != 1 else 0

    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Cohen's Kappa: {kappa_score:.4f}")
    logger.info(f"Test Confidence: {test_confidence:.3f}")
    logger.info(f"Per-class Test Acc: {[f'{acc:.3f}' for acc in per_class_test_acc]}")

    # Save final results
    if save_model_checkpoints:
        final_results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'kappa_score': float(kappa_score),
            'best_val_acc': float(best_val_acc),
            'test_confidence': float(test_confidence),
            'per_class_test_acc': [float(acc) for acc in per_class_test_acc],
            'training_completed': True,
            'total_epochs': epoch + 1,
            'total_training_time': float(total_training_time),
            'samples_per_user': samples_per_user,
            'conversion_method': conversion_method,
            'early_stopped': epochs_without_improvement >= early_stopping_patience,
            'final_lr': float(current_lr)
        }
        results_file = os.path.join(run_checkpoint_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"Training completed. All files saved to: {run_checkpoint_dir}")

    return test_acc, kappa_score