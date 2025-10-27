# =============================================
# Enhanced trainers_2d.py with Cosine Classifier, Label Smoothing, Warmup, and DROPOUT
# =============================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbones import SpectrogramResNet, LightweightSpectrogramResNet
# Import the session-based functions
from data_loader import create_session_based_dataloaders
import os
import json
from datetime import datetime
import time
import logging
import matplotlib.pyplot as plt
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# =============================================
# NEW: Cosine Classifier Head
# =============================================
class CosineClassifier(nn.Module):

    """
    Cosine similarity-based classifier head.
    Normalizes both embeddings and weights, then computes scaled cosine similarity.
    """

    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale

        # Learnable weight matrix (will be L2-normalized)
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # L2 normalize input embeddings
        x_norm = F.normalize(x, p=2, dim=1)

        # L2 normalize weight vectors
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity (dot product of normalized vectors)
        logits = F.linear(x_norm, w_norm)

        # Scale logits
        return logits * self.scale


# =============================================
# NEW: Label Smoothing Loss
# =============================================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    Prevents over-confident predictions by smoothing hard labels.
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        # pred: (batch_size, num_classes) logits
        # target: (batch_size,) class indices

        log_probs = F.log_softmax(pred, dim=1)
        num_classes = pred.size(1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# =============================================
# NEW: Warmup Scheduler
# =============================================
class WarmupScheduler:
    """
    Linear warmup followed by another scheduler.
    """

    def __init__(self, optimizer, warmup_epochs, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, *args, **kwargs):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.base_scheduler is not None:
            # Use base scheduler after warmup
            self.base_scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# =============================================
# Modified Model Wrapper with Cosine Classifier
# =============================================
class ModelWithCosineClassifier(nn.Module):
    """
    Wraps backbone and replaces final layer with cosine classifier.
    """

    def __init__(self, backbone, num_classes, embedding_dim, scale=30.0):
        super().__init__()
        self.backbone = backbone

        # Remove the final classification layer from backbone
        if hasattr(backbone, 'fc'):
            self.embedding_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            self.embedding_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        else:
            self.embedding_dim = embedding_dim

        # Add cosine classifier
        self.cosine_classifier = CosineClassifier(self.embedding_dim, num_classes, scale)

    def forward(self, x):
        embeddings = self.backbone(x)
        logits = self.cosine_classifier(embeddings)
        return logits


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

    # Annotate cells
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

    csv_path = save_path.replace(".csv", ".png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def spectrogram_trainer_2d(data_path, user_ids,
                           model_path, normalization_method='log_scale',
                           model_type='lightweight', batch_size=2, epochs=100, lr=0.001, device=None,
                           use_augmentation=False, save_model_checkpoints=True, checkpoint_every=10,
                           max_cache_size=100,
                           use_cosine_classifier=True, cosine_scale=30.0,
                           label_smoothing=0.1, warmup_epochs=5,
                           lr_scheduler_type='plateau', random_seed=None,
                           dropout_rate=0.0, classifier_dropout=0.0, weight_decay=1e-4):
    """
    Session-based PyTorch 2D spectrogram trainer with cosine classifier, label smoothing, warmup, and dropout.

    New parameters:
        use_cosine_classifier (bool): Whether to use cosine classifier instead of linear (default: True)
        cosine_scale (float): Temperature scaling for cosine classifier (default: 30.0)
        label_smoothing (float): Label smoothing factor (default: 0.1)
        warmup_epochs (int): Number of warmup epochs for learning rate (default: 5)
        lr_scheduler_type (str): Type of LR scheduler: 'plateau' or 'constant' (default: 'plateau')
        random_seed (int): Random seed for reproducibility (default: None)
        dropout_rate (float): Dropout rate for conv layers (default: 0.0)
        classifier_dropout (float): Dropout before classifier (default: 0.0)
        weight_decay (float): Weight decay for optimizer (default: 1e-4)
    """

    # Set random seeds for reproducibility
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {random_seed}")
    else:
        print("No random seed set - results may not be reproducible")

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
        os.makedirs(model_path, exist_ok=True)

        # Create unique identifier for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier_type = "cosine" if use_cosine_classifier else "linear"
        run_id = f"{normalization_method}_{model_type}_{classifier_type}_{len(user_ids)}users_{timestamp}"
        run_checkpoint_dir = os.path.join(model_path, run_id)
        os.makedirs(run_checkpoint_dir, exist_ok=True)

        # Setup logging
        logger = setup_logging(run_checkpoint_dir)
        logger.info(f"Starting 2D training run: {run_id}")
        logger.info(f"Cosine Classifier: {use_cosine_classifier}, Scale: {cosine_scale}")
        logger.info(f"Label Smoothing: {label_smoothing}, Warmup Epochs: {warmup_epochs}")
        if dropout_rate > 0 or classifier_dropout > 0:
            logger.info(f"Using Dropout - Conv layers: {dropout_rate}, Classifier: {classifier_dropout}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Model checkpoints will be saved to: {run_checkpoint_dir}")

    # --------------------------
    # Create session-based data loaders (NO DATA LEAKAGE)
    # --------------------------
    logger.info("Creating session-based data loaders (NO DATA LEAKAGE)...")
    logger.info("Split strategy: 10 sessions train, 2 sessions val, 2 sessions test per user")
    start_time = time.time()

    # Log initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

    try:
        # Use session-based splitting to avoid data leakage
        train_loader, val_loader, test_loader, sessions = create_session_based_dataloaders(
            data_path=data_path,
            user_ids=user_ids,
            normalization=normalization_method,
            batch_size=batch_size,
            augment_train=use_augmentation,
            cache_size=max_cache_size
        )

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise

    data_load_time = time.time() - start_time
    logger.info(f"Session-based data loader creation completed in {data_load_time:.2f}s")

    # Log memory usage after data loader creation
    post_loader_memory = get_memory_usage()
    logger.info(f"Memory usage after data loader creation: {post_loader_memory:.2f} GB")

    # Get dataset sizes and a sample to determine input shape
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logger.info(f"Session-based dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Get sample batch to determine input shape
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
    num_classes = len(user_ids)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes (users): {num_classes}")
    logger.info(f"Spectrogram dimensions: {input_shape[1:]} (height x width)")

    # Update DataLoader settings for memory efficiency
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
    logger.info(f"User IDs included: {user_ids[:5]}...{user_ids[-5:] if len(user_ids) > 5 else user_ids}")

    # --------------------------
    # Build model with cosine classifier and DROPOUT
    # --------------------------
    input_channels = input_shape[0]  # Should be 1 for grayscale spectrograms

    if model_type == 'lightweight':
        backbone = LightweightSpectrogramResNet(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=[32, 64, 128],
            dropout_rate=dropout_rate,
            classifier_dropout=classifier_dropout
        )
        embedding_dim = 128  # Last channel dimension
    elif model_type == 'full':
        backbone = SpectrogramResNet(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=[64, 128, 256, 512],
            dropout_rate=dropout_rate,
            classifier_dropout=classifier_dropout
        )
        embedding_dim = 512  # Last channel dimension
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap with cosine classifier if requested
    if use_cosine_classifier:
        model = ModelWithCosineClassifier(backbone, num_classes, embedding_dim, scale=cosine_scale)
        logger.info(f"Using Cosine Classifier with scale={cosine_scale}")
    else:
        model = backbone
        logger.info("Using standard Linear Classifier")

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
    # Training setup with label smoothing, warmup, and WEIGHT DECAY
    # --------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use label smoothing loss
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    logger.info(f"Using Label Smoothing Cross Entropy with smoothing={label_smoothing}")

    # Learning rate scheduler with warmup
    if lr_scheduler_type == 'plateau':
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6, verbose=True
        )
        scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, base_scheduler=base_scheduler)
        logger.info(f"Using warmup for {warmup_epochs} epochs, then ReduceLROnPlateau")
        logger.info("WARNING: ReduceLROnPlateau is non-deterministic and may affect reproducibility")
    elif lr_scheduler_type == 'constant':
        # No scheduler after warmup - just constant LR
        scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, base_scheduler=None)
        logger.info(f"Using warmup for {warmup_epochs} epochs, then constant LR (deterministic)")
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}. Use 'plateau' or 'constant'")

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
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'training_history': training_history,
            'model_config': {
                'input_channels': input_channels,
                'num_classes': num_classes,
                'num_users': len(user_ids),
                'user_ids': user_ids,
                'normalization_method': normalization_method,
                'model_type': model_type,
                'use_augmentation': use_augmentation,
                'split_type': 'session_based',
                'use_cosine_classifier': use_cosine_classifier,
                'cosine_scale': cosine_scale,
                'label_smoothing': label_smoothing,
                'warmup_epochs': warmup_epochs,
                'lr_scheduler_type': lr_scheduler_type,
                'random_seed': random_seed,
                'dropout_rate': dropout_rate,
                'classifier_dropout': classifier_dropout,
                'weight_decay': weight_decay
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
    logger.info(f"Starting session-based training for {epochs} epochs...")
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

                    # Monitor probability distribution
                    probs = F.softmax(outputs, dim=1)
                    if batch_idx % 50 == 0:
                        mean_probs = probs.mean(dim=0)
                        max_prob = mean_probs.max().item()
                        min_prob = mean_probs.min().item()
                        std_prob = probs.std(dim=1).mean().item()
                        logger.info(f"Epoch {epoch + 1} Batch {batch_idx}: "
                                    f"max_mean_prob={max_prob:.4f}, min_mean_prob={min_prob:.4f}, "
                                    f"avg_std={std_prob:.4f}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(xb)
                loss = criterion(outputs, yb)

                # Monitor probability distribution
                if batch_idx % 50 == 0:
                    probs = F.softmax(outputs, dim=1)
                    mean_probs = probs.mean(dim=0)
                    max_prob = mean_probs.max().item()
                    min_prob = mean_probs.min().item()
                    std_prob = probs.std(dim=1).mean().item()
                    logger.info(f"Epoch {epoch + 1} Batch {batch_idx}: "
                                f"max_mean_prob={max_prob:.4f}, min_mean_prob={min_prob:.4f}, "
                                f"avg_std={std_prob:.4f}")

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
        if epoch < warmup_epochs:
            scheduler.step()
        else:
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
    logger.info(f"Session-based training completed in {total_training_time:.2f}s")
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
                'early_stopped': epochs_without_improvement >= early_stopping_patience,
                'num_users': len(user_ids),
                'user_ids': user_ids,
                'split_method': 'session_based',
                'use_cosine_classifier': use_cosine_classifier,
                'cosine_scale': cosine_scale,
                'label_smoothing': label_smoothing,
                'warmup_epochs': warmup_epochs,
                'lr_scheduler_type': lr_scheduler_type,
                'random_seed': random_seed,
                'dropout_rate': dropout_rate,
                'classifier_dropout': classifier_dropout,
                'weight_decay': weight_decay
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

    logger.info("=== FINAL SESSION-BASED RESULTS (COSINE CLASSIFIER + DROPOUT) ===")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Cohen's Kappa: {kappa_score:.4f}")
    logger.info(f"Test Confidence: {test_confidence:.3f}")
    logger.info(f"Per-class Test Acc: {[f'{acc:.3f}' for acc in per_class_test_acc]}")
    logger.info(f"Users tested: {len(user_ids)} (S{user_ids[0]:03d} to S{user_ids[-1]:03d})")

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
            'num_users': len(user_ids),
            'user_ids': user_ids,
            'normalization_method': normalization_method,
            'model_type': model_type,
            'use_augmentation': use_augmentation,
            'early_stopped': epochs_without_improvement >= early_stopping_patience,
            'final_lr': float(current_lr),
            'max_cache_size': max_cache_size,
            'memory_efficient': True,
            'split_method': 'session_based',
            'data_leakage_prevented': True,
            'use_cosine_classifier': use_cosine_classifier,
            'cosine_scale': cosine_scale,
            'label_smoothing': label_smoothing,
            'warmup_epochs': warmup_epochs,
            'lr_scheduler_type': lr_scheduler_type,
            'random_seed': random_seed,
            'dropout_rate': dropout_rate,
            'classifier_dropout': classifier_dropout,
            'weight_decay': weight_decay
        }
        results_file = os.path.join(run_checkpoint_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"Session-based 2D Training completed. All files saved to: {run_checkpoint_dir}")

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
        class_names=[f"S{uid:03d}" for uid in user_ids]
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
        class_names=[f"S{uid:03d}" for uid in user_ids]
    )
    logger.info(f"Saved test confusion matrix to {test_cm_path}")

    # Final cleanup
    del all_test_outputs, all_test_targets
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return test_acc, kappa_score
