# =============================================
# Progressive Channel Training Script
# Tests classification performance with incrementally added EEG channels
# =============================================

import os
import json
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from backbones_2d import SpectrogramResNet, LightweightSpectrogramResNet
from data_loader_2d_lazy import get_file_paths_and_labels, create_memory_efficient_dataloaders
from group import group_embeddings_by_frame  # Import our custom grouping function
import matplotlib.pyplot as plt
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Configuration
OUTPUT_DIR = "/app/data/progressive_channel_results"
CHANNEL_PREFIXES = ['C2', 'C4', 'Cpz', 'Fc1', 'Iz', 'O1', 'P1', 'Po8']


def setup_logging(log_dir, run_name):
    """Setup comprehensive logging for progressive training"""
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f'progressive_training_{run_name}')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    log_file = os.path.join(log_dir, f'progressive_training_{run_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    """Save confusion matrix as both image and CSV"""
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


def prepare_channel_data(raw_embeddings_path, grouped_output_path, channel_prefixes, logger):
    """
    Prepare grouped embeddings for specific channel prefixes using our custom grouping function
    """
    logger.info(f"Preparing data for channels: {channel_prefixes}")

    # Use the group_embeddings_by_frame function from group.py
    group_embeddings_by_frame(
        embeddings_root=raw_embeddings_path,
        output_root=grouped_output_path,
        include_prefixes=channel_prefixes,
        verbose=True
    )

    logger.info(f"Data preparation completed for channels: {channel_prefixes}")


def progressive_channel_trainer(raw_embeddings_path, user_ids, normalization_method='log_scale',
                                model_type='lightweight', batch_size=16, epochs=100, lr=0.001,
                                device=None, use_augmentation=False, max_cache_size=100,
                                checkpoint_every=10):
    """
    Progressive channel training that incrementally adds EEG channels.

    Args:
        raw_embeddings_path: Path to raw embedding files (before grouping)
        user_ids: List of user IDs for classification
        normalization_method: Normalization method for spectrograms
        model_type: Type of model ('lightweight' or 'full')
        batch_size: Training batch size
        epochs: Number of training epochs per channel combination
        lr: Learning rate
        device: Training device ('cuda' or 'cpu')
        use_augmentation: Whether to use data augmentation
        max_cache_size: Maximum cache size for memory efficiency
        checkpoint_every: Save checkpoint every N epochs
    """

    if device is None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                device = 'cuda'
            except RuntimeError:
                print("Warning: CUDA available but not working, falling back to CPU")
                device = 'cpu'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(OUTPUT_DIR, f"progressive_channels_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Setup main logger
    main_logger = setup_logging(main_output_dir, "main")
    main_logger.info(f"Starting progressive channel training")
    main_logger.info(f"Channel progression: {CHANNEL_PREFIXES}")
    main_logger.info(f"Results will be saved to: {main_output_dir}")

    # Store results for each channel combination
    progressive_results = {
        'channel_combinations': [],
        'test_accuracies': [],
        'kappa_scores': [],
        'best_val_accuracies': [],
        'training_times': [],
        'total_parameters': [],
        'configuration': {
            'normalization_method': normalization_method,
            'model_type': model_type,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'use_augmentation': use_augmentation,
            'max_cache_size': max_cache_size,
            'user_ids': user_ids,
            'device': device
        }
    }

    # Progressive training loop
    for i in range(1, len(CHANNEL_PREFIXES) + 1):
        current_channels = CHANNEL_PREFIXES[:i]
        channel_str = "_".join(current_channels)

        main_logger.info(f"\n{'=' * 60}")
        main_logger.info(f"TRAINING WITH CHANNELS: {current_channels} ({i}/{len(CHANNEL_PREFIXES)})")
        main_logger.info(f"{'=' * 60}")

        # Create specific output directory for this channel combination
        channel_output_dir = os.path.join(main_output_dir, f"channels_{channel_str}")
        grouped_data_dir = os.path.join(channel_output_dir, "grouped_data")
        os.makedirs(channel_output_dir, exist_ok=True)

        # Setup logger for this specific training
        channel_logger = setup_logging(channel_output_dir, channel_str)

        try:
            # Step 1: Prepare grouped data for current channel combination
            channel_logger.info("Step 1: Preparing grouped embeddings...")
            prepare_channel_data(raw_embeddings_path, grouped_data_dir, current_channels, channel_logger)

            # Step 2: Train model with current channel combination
            channel_logger.info("Step 2: Starting model training...")
            start_time = time.time()

            test_acc, kappa_score = train_single_channel_combination(
                data_path=grouped_data_dir,
                user_ids=user_ids,
                channel_combination=current_channels,
                output_dir=channel_output_dir,
                normalization_method=normalization_method,
                model_type=model_type,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                use_augmentation=use_augmentation,
                max_cache_size=max_cache_size,
                checkpoint_every=checkpoint_every,
                logger=channel_logger
            )

            training_time = time.time() - start_time

            # Store results
            progressive_results['channel_combinations'].append(current_channels.copy())
            progressive_results['test_accuracies'].append(float(test_acc))
            progressive_results['kappa_scores'].append(float(kappa_score))
            progressive_results['training_times'].append(float(training_time))

            channel_logger.info(f"Completed training for {current_channels}")
            channel_logger.info(f"Test Accuracy: {test_acc:.4f}, Kappa: {kappa_score:.4f}")
            channel_logger.info(f"Training Time: {training_time:.2f}s")

            # Save intermediate results
            intermediate_results_file = os.path.join(main_output_dir, 'progressive_results_intermediate.json')
            with open(intermediate_results_file, 'w') as f:
                json.dump(progressive_results, f, indent=2)

        except Exception as e:
            main_logger.error(f"Error training with channels {current_channels}: {e}")
            # Continue with next combination instead of failing completely
            progressive_results['channel_combinations'].append(current_channels.copy())
            progressive_results['test_accuracies'].append(0.0)
            progressive_results['kappa_scores'].append(0.0)
            progressive_results['training_times'].append(0.0)
            continue

    # Final results analysis
    main_logger.info(f"\n{'=' * 60}")
    main_logger.info("PROGRESSIVE TRAINING COMPLETED - FINAL RESULTS")
    main_logger.info(f"{'=' * 60}")

    for i, (channels, acc, kappa, time_taken) in enumerate(zip(
            progressive_results['channel_combinations'],
            progressive_results['test_accuracies'],
            progressive_results['kappa_scores'],
            progressive_results['training_times']
    )):
        main_logger.info(f"Channels {i + 1:2d}: {str(channels):30s} | "
                         f"Acc: {acc:.4f} | "
                         f"Kappa: {kappa:.4f} | "
                         f"Time: {time_taken:6.1f}s")

    # Find best performing combination
    if progressive_results['test_accuracies']:
        best_idx = np.argmax(progressive_results['test_accuracies'])
        best_channels = progressive_results['channel_combinations'][best_idx]
        best_acc = progressive_results['test_accuracies'][best_idx]
        main_logger.info(f"\nBest performing combination: {best_channels} with accuracy {best_acc:.4f}")

    # Save final results
    final_results_file = os.path.join(main_output_dir, 'progressive_results_final.json')
    with open(final_results_file, 'w') as f:
        json.dump(progressive_results, f, indent=2)

    # Create summary plot
    create_progressive_summary_plot(progressive_results, main_output_dir)

    main_logger.info(f"\nAll results saved to: {main_output_dir}")

    return progressive_results


def train_single_channel_combination(data_path, user_ids, channel_combination, output_dir,
                                     normalization_method, model_type, batch_size, epochs, lr,
                                     device, use_augmentation, max_cache_size, checkpoint_every, logger):
    """
    Train a single model with a specific channel combination.
    This is essentially the same as spectrogram_trainer_2d but adapted for our use case.
    """

    # Log initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

    # Create data loaders
    logger.info("Creating memory-efficient data loaders...")
    try:
        train_loader, val_loader, test_loader, sessions = create_memory_efficient_dataloaders(
            data_path=data_path,
            user_ids=user_ids,
            samples_per_user=-1,  # Use all available samples (100%)
            normalization=normalization_method,
            batch_size=batch_size,
            augment_train=use_augmentation,
            cache_size=max_cache_size
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise

    # Get dataset information
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Get sample to determine input shape
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
    num_classes = len(user_ids)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Channel combination: {channel_combination}")

    # Update DataLoaders for memory efficiency
    train_loader = DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    val_loader = DataLoader(
        val_loader.dataset, batch_size=batch_size,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    test_loader = DataLoader(
        test_loader.dataset, batch_size=batch_size,
        num_workers=0, pin_memory=False, persistent_workers=False
    )

    # Build model
    input_channels = input_shape[0]

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
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: {model_type}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6
    )

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    early_stopping_patience = 25

    training_history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'learning_rate': [], 'per_class_train_acc': [], 'per_class_val_acc': [],
        'train_confidence': [], 'val_confidence': [], 'epochs_completed': 0
    }

    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
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
            all_train_outputs.append(outputs.detach().cpu())
            all_train_targets.append(yb.detach().cpu())

            # Periodic memory cleanup
            if batch_idx % 20 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

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
                all_val_outputs.append(outputs.cpu())
                all_val_targets.append(yb.cpu())

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
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        current_memory = get_memory_usage()

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            logger.info(f"Epoch {epoch + 1:3d}/{epochs} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"Memory: {current_memory:.1f}GB | "
                        f"LR: {current_lr:.2e} | "
                        f"Train Loss: {epoch_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f} | "
                        f"Best Val: {best_val_acc:.4f}" +
                        (" NEW BEST" if is_best else ""))

        # Plot training curves every 25 epochs
        if (epoch + 1) % 25 == 0:
            plot_path = os.path.join(output_dir, f'training_curves_epoch_{epoch + 1}.png')
            plot_training_curves(training_history, plot_path)

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

        # Memory cleanup at end of epoch
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model for final evaluation")

    # Final evaluation on test set
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
            all_test_outputs.append(outputs.cpu())
            all_test_targets.append(yb.cpu())

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
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(all_preds, minlength=num_classes)
    pe = np.sum(true_counts * pred_counts) / (len(y_true) ** 2)
    kappa_score = (po - pe) / (1 - pe) if pe < 1 else 0.0

    # Log final results
    total_training_time = time.time() - training_start_time

    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Cohen's Kappa: {kappa_score:.4f}")
    logger.info(f"Test Confidence: {test_confidence:.3f}")
    logger.info(f"Per-class Test Acc: {[f'{acc:.3f}' for acc in per_class_test_acc]}")
    logger.info(f"Total Training Time: {total_training_time:.2f}s")

    # Save final results for this channel combination
    channel_results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'kappa_score': float(kappa_score),
        'best_val_acc': float(best_val_acc),
        'test_confidence': float(test_confidence),
        'per_class_test_acc': [float(acc) for acc in per_class_test_acc],
        'training_completed': True,
        'total_epochs': epoch + 1,
        'total_training_time': float(total_training_time),
        'channel_combination': channel_combination,
        'normalization_method': normalization_method,
        'model_type': model_type,
        'use_augmentation': use_augmentation,
        'early_stopped': epochs_without_improvement >= early_stopping_patience,
        'final_lr': float(current_lr),
        'max_cache_size': max_cache_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }

    results_file = os.path.join(output_dir, 'channel_results.json')
    with open(results_file, 'w') as f:
        json.dump(channel_results, f, indent=2)

    # Save training history
    history_file = os.path.join(output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
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
    final_plot_path = os.path.join(output_dir, 'final_training_curves.png')
    plot_training_curves(training_history, final_plot_path)

    # Generate confusion matrices
    logger.info("Generating confusion matrices...")

    # Training confusion matrix
    all_train_preds, all_train_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().tolist())
            all_train_labels.extend(labels.cpu().tolist())

    train_cm_path = os.path.join(output_dir, "train_confusion_matrix.png")
    save_confusion_matrix_torch(
        all_train_labels, all_train_preds, num_classes=num_classes,
        save_path=train_cm_path, class_names=user_ids
    )

    # Test confusion matrix
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().tolist())
            all_test_labels.extend(labels.cpu().tolist())

    test_cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    save_confusion_matrix_torch(
        all_test_labels, all_test_preds, num_classes=num_classes,
        save_path=test_cm_path, class_names=user_ids
    )

    # Final cleanup
    del all_test_outputs, all_test_targets
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Training completed for channels: {channel_combination}")

    return test_acc, kappa_score


def create_progressive_summary_plot(results, output_dir):
    """Create summary visualization of progressive channel training results"""

    channel_combinations = results['channel_combinations']
    test_accuracies = results['test_accuracies']
    kappa_scores = results['kappa_scores']

    # Create channel labels for x-axis
    channel_labels = []
    for channels in channel_combinations:
        if len(channels) == 1:
            channel_labels.append(channels[0])
        else:
            channel_labels.append("+".join(channels))

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Test Accuracy
    x_pos = np.arange(len(channel_labels))
    bars1 = ax1.bar(x_pos, test_accuracies, alpha=0.8, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Channel Combinations')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy vs Channel Combinations')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.0])

    # Add value labels on bars
    for bar, acc in zip(bars1, test_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Cohen's Kappa
    bars2 = ax2.bar(x_pos, kappa_scores, alpha=0.8, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Channel Combinations')
    ax2.set_ylabel("Cohen's Kappa")
    ax2.set_title("Cohen's Kappa vs Channel Combinations")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])

    # Add value labels on bars
    for bar, kappa in zip(bars2, kappa_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{kappa:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'progressive_channel_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a line plot showing the progression
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(range(len(test_accuracies)), test_accuracies, 'o-',
            linewidth=2, markersize=8, color='steelblue', label='Test Accuracy')
    ax.plot(range(len(kappa_scores)), kappa_scores, 's--',
            linewidth=2, markersize=8, color='lightcoral', label="Cohen's Kappa")

    ax.set_xlabel('Progressive Channel Combinations')
    ax.set_ylabel('Score')
    ax.set_title('Progressive Channel Training Results')
    ax.set_xticks(range(len(channel_labels)))
    ax.set_xticklabels(channel_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.0])

    # Add annotations for peak values
    if test_accuracies:
        best_acc_idx = np.argmax(test_accuracies)
        ax.annotate(f'Best Acc: {test_accuracies[best_acc_idx]:.3f}',
                    xy=(best_acc_idx, test_accuracies[best_acc_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='steelblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    if kappa_scores:
        best_kappa_idx = np.argmax(kappa_scores)
        ax.annotate(f'Best Kappa: {kappa_scores[best_kappa_idx]:.3f}',
                    xy=(best_kappa_idx, kappa_scores[best_kappa_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Save the line plot
    line_plot_path = os.path.join(output_dir, 'progressive_channel_trends.png')
    plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main function to run progressive channel training
    Usage example and configuration
    """

    # Configuration - modify these paths and parameters as needed
    config = {
        'raw_embeddings_path': '/app/logs/eval_embeddings',  # Path to raw embeddings before grouping
        'user_ids': ['user1', 'user2', 'user3', 'user4', 'user5'],  # Replace with actual user IDs
        'normalization_method': 'log_scale',
        'model_type': 'lightweight',  # or 'full'
        'batch_size': 16,
        'epochs': 100,
        'lr': 0.001,
        'device': None,  # Will auto-detect CUDA/CPU
        'use_augmentation': False,
        'max_cache_size': 100,
        'checkpoint_every': 10
    }

    print("Starting Progressive Channel Training")
    print("=" * 50)
    print(f"Channel progression: {CHANNEL_PREFIXES}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print("=" * 50)

    # Run progressive training
    results = progressive_channel_trainer(**config)

    print("\nProgressive training completed!")
    print("Check the output directory for detailed results and visualizations.")

    return results


if __name__ == "__main__":
    # Run the progressive training
    results = main()