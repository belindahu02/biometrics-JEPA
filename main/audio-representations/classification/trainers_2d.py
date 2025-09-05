# =============================================
# Enhanced trainers_2d.py with Memory-Efficient Loading
# =============================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from backbones_2d import SpectrogramResNet, LightweightSpectrogramResNet
# Import the new memory-efficient functions
from data_loader_2d_lazy import get_file_paths_and_labels, create_memory_efficient_dataloaders
import os
import json
from datetime import datetime
import time
import logging
import matplotlib.pyplot as plt
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

OUTPUT_DIR="/app/data/model_checkpoints_2d"

def setup_logging(log_dir):
    """Setup comprehensive logging for training"""
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('training_2d')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    log_file = os.path.join(log_dir, f'training_2d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    """Plot and save training curves for 2D training"""
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


def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb / 1024  # Convert to GB
    except ImportError:
        return 0.0  # Return 0 if psutil not available

def save_confusion_matrix_torch(y_true, y_pred, num_classes, save_path, class_names=None):
    # Initialize confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm = cm.numpy()

    # Save raw confusion matrix as CSV for later use
    csv_path = save_path.replace(".png", ".csv")
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    # Normalize per row (to show proportions)
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    # Tick marks
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Labels
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Annotate only non-zero cells
    thresh = cm_normalized.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                        fontsize=6)

    step = max(1, num_classes // 20)  # show ~20 ticks max
    ax.set_xticks(np.arange(0, num_classes, step))
    ax.set_yticks(np.arange(0, num_classes, step))
    ax.set_xticklabels([class_names[i] for i in range(0, num_classes, step)], rotation=45, ha="right")
    ax.set_yticklabels([class_names[i] for i in range(0, num_classes, step)])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def spectrogram_trainer_2d(samples_per_user, data_path, user_ids, normalization_method='log_scale',
                           model_type='lightweight', batch_size=16, epochs=100, lr=0.001, device=None,
                           use_augmentation=False, save_model_checkpoints=True, checkpoint_every=10,
                           max_cache_size=100):
    """
    Memory-efficient PyTorch 2D spectrogram trainer.

    NEW PARAMETERS:
        max_cache_size: Maximum number of spectrograms to keep in memory cache (default: 100)
    """
    if device is None:
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                torch.cuda.empty_cache()
                device = 'cuda'
            except RuntimeError:
                print("Warning: CUDA available but not working, falling back to CPU")
                device = 'cpu'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Force CPU if CUDA is requested but not available
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = 'cpu'

    # Create checkpoint directory
    if save_model_checkpoints:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Create unique identifier for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{normalization_method}_{model_type}_{samples_per_user}samples_{timestamp}"
        run_checkpoint_dir = os.path.join(OUTPUT_DIR, run_id)
        os.makedirs(run_checkpoint_dir, exist_ok=True)

        # Setup logging
        logger = setup_logging(run_checkpoint_dir)
        logger.info(f"Starting 2D training run: {run_id}")
        logger.info(f"Model checkpoints will be saved to: {run_checkpoint_dir}")
    else:
        logger = setup_logging("./logs_2d")

    # --------------------------
    # Create memory-efficient data loaders
    # --------------------------
    logger.info("Creating memory-efficient data loaders...")
    start_time = time.time()

    # Log initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

    try:
        train_loader, val_loader, test_loader, sessions = create_memory_efficient_dataloaders(
            data_path=data_path,
            user_ids=user_ids,
            samples_per_user=samples_per_user,
            normalization=normalization_method,
            batch_size=batch_size,
            augment_train=use_augmentation,
            cache_size=max_cache_size
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise

    data_load_time = time.time() - start_time
    logger.info(f"Data loader creation completed in {data_load_time:.2f}s")

    # Log memory usage after data loader creation
    post_loader_memory = get_memory_usage()
    logger.info(f"Memory usage after data loader creation: {post_loader_memory:.2f} GB")

    # Get dataset sizes and a sample to determine input shape
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Get sample batch to determine input shape
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
    num_classes = len(user_ids)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Spectrogram dimensions: {input_shape[1:]} (height x width)")

    # Convert to PyTorch tensors - THIS IS NO LONGER NEEDED since the dataset handles it
    # The data loaders already return tensors

    # Create DataLoaders - ALREADY DONE ABOVE
    # Update settings for memory efficiency
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Reduced from 4 to avoid memory issues
        pin_memory=False,  # Disabled to save memory
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    logger.info(f"Batch size: {batch_size}, Total batches per epoch: {len(train_loader)}")
    logger.info(f"Using augmentation: {use_augmentation}")
    logger.info(f"Cache size: {max_cache_size}")

    # --------------------------
    # Build model
    # --------------------------
    input_channels = input_shape[0]  # Should be 1 for grayscale spectrograms

    if model_type == 'lightweight':
        model = LightweightSpectrogramResNet(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=[32, 64, 128]
        ).to(device)
    elif model_type == 'full':
        model = SpectrogramResNet(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=[64, 128, 256, 512]
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: {model_type}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --------------------------
    # Training setup
    # --------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
                'input_channels': input_channels,
                'num_classes': num_classes,
                'samples_per_user': samples_per_user,
                'normalization_method': normalization_method,
                'model_type': model_type,
                'use_augmentation': use_augmentation
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

    # --------------------------
    # Training loop with memory management
    # --------------------------
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Early stopping patience: {early_stopping_patience}")

    training_start_time = time.time()
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        all_train_outputs = []
        all_train_targets = []

        # Memory cleanup before epoch
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler and device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * xb.size(0)
            all_train_outputs.append(outputs.detach().cpu())  # Move to CPU to save GPU memory
            all_train_targets.append(yb.detach().cpu())

            # Periodic memory cleanup
            if batch_idx % 20 == 0:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

                # Log memory usage periodically
                if batch_idx % 100 == 0:
                    current_memory = get_memory_usage()
                    logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}: Memory usage: {current_memory:.2f} GB")

        # Calculate training metrics
        all_train_outputs = torch.cat(all_train_outputs).to(device)
        all_train_targets = torch.cat(all_train_targets).to(device)
        train_acc, per_class_train_acc, train_confidence = calculate_metrics(all_train_outputs, all_train_targets)
        epoch_loss = running_loss / train_size

        # Clear training outputs from memory
        del all_train_outputs, all_train_targets
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_outputs = []
        all_val_targets = []

        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(val_loader):
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                all_val_outputs.append(outputs.cpu())  # Move to CPU
                all_val_targets.append(yb.cpu())

                # Memory cleanup during validation too
                if batch_idx % 20 == 0 and device == 'cuda':
                    torch.cuda.empty_cache()

        all_val_outputs = torch.cat(all_val_outputs).to(device)
        all_val_targets = torch.cat(all_val_targets).to(device)
        val_acc, per_class_val_acc, val_confidence = calculate_metrics(all_val_outputs, all_val_targets)
        val_loss /= val_size

        # Clear validation outputs from memory
        del all_val_outputs, all_val_targets
        if device == 'cuda':
            torch.cuda.empty_cache()

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
            best_model_state = model.state_dict().copy()  # Make a copy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        current_memory = get_memory_usage()

        # Detailed logging every epoch
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            logger.info(f"Epoch {epoch + 1:3d}/{epochs} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"Memory: {current_memory:.1f}GB | "
                        f"LR: {current_lr:.2e} | "
                        f"Train Loss: {epoch_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f} | "
                        f"Train Conf: {train_confidence:.3f} | "
                        f"Val Conf: {val_confidence:.3f} | "
                        f"Best Val: {best_val_acc:.4f}" +
                        (" NEW BEST" if is_best else ""))

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

        # Memory cleanup at end of epoch
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

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

    # Evaluation on test set with memory management
    logger.info("Starting final evaluation on test set...")
    model.eval()
    test_loss = 0.0
    all_test_outputs = []
    all_test_targets = []

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(test_loader):
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item() * xb.size(0)
            all_test_outputs.append(outputs.cpu())  # Move to CPU
            all_test_targets.append(yb.cpu())

            # Memory cleanup during test evaluation
            if batch_idx % 20 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

    all_test_outputs = torch.cat(all_test_outputs).to(device)
    all_test_targets = torch.cat(all_test_targets).to(device)
    test_acc, per_class_test_acc, test_confidence = calculate_metrics(all_test_outputs, all_test_targets)
    test_loss /= test_size

    # Compute Cohen's Kappa
    all_preds = all_test_outputs.argmax(dim=1).cpu().numpy().astype(int)
    y_true = all_test_targets.cpu().numpy().astype(int)

    po = np.mean(all_preds == y_true)

    # Force both vectors to length = num_classes
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(all_preds, minlength=num_classes)

    pe = np.sum(true_counts * pred_counts) / (len(y_true) ** 2)
    kappa_score = (po - pe) / (1 - pe) if pe < 1 else 0.0

    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Cohen's Kappa: {kappa_score:.4f}")
    logger.info(f"Test Confidence: {test_confidence:.3f}")
    logger.info(f"Per-class Test Acc: {[f'{acc:.3f}' for acc in per_class_test_acc]}")

    # Log final memory usage
    final_memory = get_memory_usage()
    logger.info(f"Final memory usage: {final_memory:.2f} GB")

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
            'normalization_method': normalization_method,
            'model_type': model_type,
            'use_augmentation': use_augmentation,
            'early_stopped': epochs_without_improvement >= early_stopping_patience,
            'final_lr': float(current_lr),
            'max_cache_size': max_cache_size,
            'memory_efficient': True
        }
        results_file = os.path.join(run_checkpoint_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"2D Training completed. All files saved to: {run_checkpoint_dir}")

    # --------------------------
    # Confusion Matrices (Train + Test)
    # --------------------------
    logger.info("Generating confusion matrices...")

    # --- Training confusion matrix ---
    all_train_preds, all_train_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().tolist())
            all_train_labels.extend(labels.cpu().tolist())

    train_cm_path = os.path.join(run_checkpoint_dir, "train_confusion_matrix.png")
    save_confusion_matrix_torch(
        all_train_labels,
        all_train_preds,
        num_classes=num_classes,
        save_path=train_cm_path,
        class_names=user_ids
    )
    logger.info(f"Saved training confusion matrix to {train_cm_path}")

    # --- Test confusion matrix ---
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().tolist())
            all_test_labels.extend(labels.cpu().tolist())

    test_cm_path = os.path.join(run_checkpoint_dir, "test_confusion_matrix.png")
    save_confusion_matrix_torch(
        all_test_labels,
        all_test_preds,
        num_classes=num_classes,
        save_path=test_cm_path,
        class_names=user_ids
    )
    logger.info(f"Saved test confusion matrix to {test_cm_path}")

    # Final cleanup
    del all_test_outputs, all_test_targets
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return test_acc, kappa_score


