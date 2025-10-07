"""
Specialized data loader for masking experiment with proper session-based splitting
Handles training on sessions 1-10, validation on 11-12, and testing on 13-14
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torch.nn as nn


class SessionBasedMaskingDataset(Dataset):
    """Dataset that loads data from specific sessions only"""

    def __init__(self, data_dir, user_ids, session_nums, add_channel_dim=True):
        self.data_dir = Path(data_dir)
        self.user_ids = user_ids
        self.session_nums = session_nums
        self.add_channel_dim = add_channel_dim

        self.file_paths = []
        self.labels = []

        self._load_session_files()

        print(f"SessionBasedMaskingDataset created:")
        print(f"  Users: S{min(user_ids):03d}-S{max(user_ids):03d} ({len(user_ids)} users)")
        print(f"  Sessions: {session_nums}")
        print(f"  Total files: {len(self.file_paths)}")

    def _load_session_files(self):
        """Load files from specified sessions only"""
        for user_idx, user_id in enumerate(self.user_ids):
            user_folder = f"S{user_id:03d}"
            user_path = self.data_dir / user_folder

            if not user_path.exists():
                print(f"Warning: User directory {user_path} not found")
                continue

            files_found = 0
            for session_num in self.session_nums:
                session_folder = f"{user_folder}R{session_num:02d}"
                session_path = user_path / session_folder

                if not session_path.exists():
                    print(f"Warning: Session directory {session_path} not found")
                    continue

                # Look for grouped/stacked files
                session_files = list(session_path.glob("*_stacked.npy"))

                for file_path in session_files:
                    # Verify file is valid
                    try:
                        # Quick check - just load header to verify it's readable
                        with open(file_path, 'rb') as f:
                            np.lib.format.read_magic(f)
                            shape, _, _ = np.lib.format.read_array_header_1_0(f)

                        self.file_paths.append(file_path)
                        self.labels.append(user_idx)
                        files_found += 1

                    except Exception as e:
                        print(f"Warning: Invalid file {file_path}: {e}")

            if files_found == 0:
                print(f"Warning: No valid files found for user S{user_id:03d}")
            else:
                print(f"User S{user_id:03d}: {files_found} files loaded from sessions {self.session_nums}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load spectrogram data
            data = np.load(file_path)

            # Handle different data shapes
            if data.ndim == 2:
                # (height, width) -> (1, height, width)
                if self.add_channel_dim:
                    data = data[np.newaxis, :, :]
            elif data.ndim == 3:
                # Already has channel dimension
                pass
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            # Validate data
            if not np.isfinite(data).all():
                print(f"Warning: Non-finite values in {file_path}")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor as fallback
            if self.add_channel_dim:
                fallback_data = torch.zeros((1, 80, 2000), dtype=torch.float32)  # typical spectrogram size
            else:
                fallback_data = torch.zeros((80, 2000), dtype=torch.float32)
            return fallback_data, torch.tensor(label, dtype=torch.long)


def create_masking_experiment_dataloaders(data_dir, user_ids, batch_size=16):
    """
    Create data loaders for masking experiment with proper session splitting:
    - Train: sessions 1-10
    - Validation: sessions 11-12
    - Test: sessions 13-14
    """
    print("Creating masking experiment data loaders...")
    print(f"Data directory: {data_dir}")
    print(f"Users: S{min(user_ids):03d}-S{max(user_ids):03d} ({len(user_ids)} users)")

    # Define session splits
    train_sessions = list(range(1, 11))  # Sessions 1-10
    val_sessions = [11, 12]  # Sessions 11-12
    test_sessions = [13, 14]  # Sessions 13-14

    print(f"Session split: Train={train_sessions}, Val={val_sessions}, Test={test_sessions}")

    # Create datasets for each split
    train_dataset = SessionBasedMaskingDataset(
        data_dir=data_dir,
        user_ids=user_ids,
        session_nums=train_sessions,
        add_channel_dim=True
    )

    val_dataset = SessionBasedMaskingDataset(
        data_dir=data_dir,
        user_ids=user_ids,
        session_nums=val_sessions,
        add_channel_dim=True
    )

    test_dataset = SessionBasedMaskingDataset(
        data_dir=data_dir,
        user_ids=user_ids,
        session_nums=test_sessions,
        add_channel_dim=True
    )

    # Verify we have data for all splits
    if len(train_dataset) == 0:
        raise ValueError("No training data found!")
    if len(val_dataset) == 0:
        print("Warning: No validation data found!")
    if len(test_dataset) == 0:
        print("Warning: No test data found!")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    # Verify class distribution
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    unique_labels, counts = np.unique(train_labels, return_counts=True)

    print(f"Training class distribution:")
    for label, count in zip(unique_labels, counts):
        user_id = user_ids[label]
        print(f"  S{user_id:03d}: {count} samples")

    if len(unique_labels) != len(user_ids):
        missing_users = set(range(len(user_ids))) - set(unique_labels)
        print(f"WARNING: Missing training data for users: {[f'S{user_ids[i]:03d}' for i in missing_users]}")

    return train_loader, val_loader, test_loader


class MaskingExperimentTrainer:
    """Simplified trainer specifically for masking experiment"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Train model and return best validation accuracy"""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=10, min_lr=1e-6
        )

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 20

        print(f"Training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            train_acc = train_correct / train_total

            # Validation phase
            if len(val_loader) > 0:
                val_acc = self.evaluate_model(val_loader)
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Periodic logging
                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1:3d}: Train={train_acc:.4f}, Val={val_acc:.4f}, LR={current_lr:.2e}")

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                # No validation data
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1:3d}: Train={train_acc:.4f}")
                best_val_acc = train_acc
                best_model_state = self.model.state_dict().copy()

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return best_val_acc

    def evaluate_model(self, data_loader):
        """Evaluate model and return accuracy"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total if total > 0 else 0.0

    def evaluate_with_kappa(self, data_loader):
        """Evaluate model and return accuracy and Cohen's kappa"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        if len(all_targets) == 0:
            return 0.0, 0.0

        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

        # Calculate Cohen's kappa
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(all_targets, all_preds)

        return accuracy, kappa

    def save_model(self, filepath):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__
        }, filepath)

    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def test_masking_data_loader():
    """Test the masking data loader"""
    # Test configuration
    test_data_dir = "/app/data/grouped_embeddings_full"  # Adjust path as needed
    test_user_ids = list(range(1, 6))  # Test with first 5 users

    print("Testing masking data loader...")

    try:
        train_loader, val_loader, test_loader = create_masking_experiment_dataloaders(
            data_dir=test_data_dir,
            user_ids=test_user_ids,
            batch_size=4
        )

        print("Data loaders created successfully!")

        # Test loading a batch from each
        if len(train_loader) > 0:
            train_batch = next(iter(train_loader))
            print(f"Train batch: inputs={train_batch[0].shape}, labels={train_batch[1].shape}")
            print(f"Train labels in batch: {train_batch[1].tolist()}")

        if len(val_loader) > 0:
            val_batch = next(iter(val_loader))
            print(f"Val batch: inputs={val_batch[0].shape}, labels={val_batch[1].shape}")

        if len(test_loader) > 0:
            test_batch = next(iter(test_loader))
            print(f"Test batch: inputs={test_batch[0].shape}, labels={test_batch[1].shape}")

        print("✅ Masking data loader test passed!")
        return True

    except Exception as e:
        print(f"❌ Masking data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_masking_data_loader()