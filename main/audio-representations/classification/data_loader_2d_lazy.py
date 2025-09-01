# =============================================
# FIXED data_loader_2d_lazy.py - No Double Normalization
# =============================================

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class SpectrogramDataset(Dataset):
    """
    Fixed dataset that handles pre-normalized spectrograms correctly
    """

    def __init__(self, file_paths, labels, normalization='none', add_channel_dim=True,
                 augment=False, cache_size=100):
        self.file_paths = file_paths
        self.labels = labels
        self.normalization = normalization
        self.add_channel_dim = add_channel_dim
        self.augment = augment

        # Simple LRU cache
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        # Check if data is already normalized
        self._check_data_normalization()

    def _check_data_normalization(self):
        """Check if data is already normalized by sampling a few files"""
        print("Checking if data is pre-normalized...")

        sample_indices = np.random.choice(len(self.file_paths), min(5, len(self.file_paths)), replace=False)
        means = []
        stds = []
        ranges = []

        for idx in sample_indices:
            try:
                spec = np.load(self.file_paths[idx])
                means.append(np.mean(spec))
                stds.append(np.std(spec))
                ranges.append((np.min(spec), np.max(spec)))
            except:
                continue

        if means:
            avg_mean = np.mean(means)
            avg_std = np.mean(stds)
            overall_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))

            print(f"Data check - Mean: {avg_mean:.4f}, Std: {avg_std:.4f}, Range: {overall_range}")

            # Check if data appears pre-normalized (mean~0, std~1)
            if abs(avg_mean) < 0.1 and 0.5 < avg_std < 2.0:
                print("✅ Data appears to be PRE-NORMALIZED")
                if self.normalization != 'none':
                    print(
                        f"⚠️  WARNING: You requested '{self.normalization}' normalization on already normalized data!")
                    print("   This will cause double normalization and gradient explosion.")
                    print("   Switching to 'none' normalization...")
                    self.normalization = 'none'
            else:
                print("Data appears to need normalization")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Check cache first
        if file_path in self.cache:
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            spec = self.cache[file_path].copy()
        else:
            # Load from disk
            spec = np.load(file_path)

            # Validate data
            if not np.isfinite(spec).all():
                print(f"Warning: Non-finite values in {file_path}")
                spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

            # Add to cache
            if len(self.cache) >= self.cache_size:
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]

            self.cache[file_path] = spec.copy()
            self.cache_order.append(file_path)

        # Apply normalization ONLY if data is not pre-normalized
        if self.normalization and self.normalization != 'none':
            spec = self._normalize_single(spec, self.normalization)

        # Add channel dimension if needed
        if self.add_channel_dim:
            spec = spec[np.newaxis, :, :]

        # Apply augmentation if training
        if self.augment:
            spec = self._augment_single(spec)

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _normalize_single(self, spec, method):
        """
        SAFE normalization methods that won't cause gradient explosion
        """
        if method == 'none':
            return spec

        elif method == 'safe_clip':
            # Just clip extreme values without changing distribution
            return np.clip(spec, -5.0, 5.0)

        elif method == 'robust_scale':
            # Use percentiles instead of min/max for robustness
            p5 = np.percentile(spec, 5)
            p95 = np.percentile(spec, 95)
            if p95 - p5 > 1e-8:
                spec = (spec - p5) / (p95 - p5)
                spec = np.clip(spec, 0, 1)
            return spec

        elif method == 'gentle_z_score':
            # Z-score with clipping
            mean = np.mean(spec)
            std = np.std(spec)
            if std < 1e-8:
                return np.zeros_like(spec)
            else:
                result = (spec - mean) / std
                return np.clip(result, -3.0, 3.0)  # Gentle clipping

        # AVOID log_scale, decibel, or any extreme transformations on pre-normalized data
        return spec

    def _augment_single(self, spec):
        """Gentle augmentation that won't break pre-normalized data"""
        if np.random.rand() > 0.8:  # Very light augmentation
            if spec.ndim == 3:  # (channel, time, freq)
                t_max = spec.shape[1]
                mask_size = np.random.randint(1, min(t_max // 30, 20))  # Small masks
                start = np.random.randint(0, max(1, t_max - mask_size))
                # Use noise instead of zeros to maintain distribution
                spec[:, start:start + mask_size, :] *= np.random.uniform(0.1, 0.3)
            else:
                t_max = spec.shape[0]
                mask_size = np.random.randint(1, min(t_max // 30, 20))
                start = np.random.randint(0, max(1, t_max - mask_size))
                spec[start:start + mask_size, :] *= np.random.uniform(0.1, 0.3)

        return spec


def get_file_paths_and_labels(data_path, user_ids):
    """
    Get file paths and labels without loading the actual data.
    Returns paths and corresponding user labels.
    """
    file_paths = []
    labels = []
    sessions_info = []

    for user_idx, user_id in enumerate(user_ids):
        user_folder = f"S{user_id:03d}"
        user_path = os.path.join(data_path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: Path {user_path} does not exist")
            sessions_info.append(0)
            continue

        session_count = 0
        user_files = []

        # Loop over session folders
        for item in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, item)
            if os.path.isdir(session_path) and item.startswith(user_folder + 'R'):
                for npy_file in sorted(os.listdir(session_path)):
                    if npy_file.endswith("_stacked.npy"):
                        spec_file = os.path.join(session_path, npy_file)
                        try:
                            # Load just the header to check shape
                            with open(spec_file, 'rb') as f:
                                header = np.lib.format.read_magic(f)
                                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                                if shape in [(4160, 768), (520, 768)]:
                                    user_files.append(spec_file)
                                    session_count += 1
                        except Exception as e:
                            print(f"Error checking {spec_file}: {e}")

        file_paths.extend(user_files)
        labels.extend([user_idx] * len(user_files))
        sessions_info.append(session_count)
        print(f"User {user_id}: {len(user_files)} spectrograms found")

    return file_paths, labels, sessions_info


def create_memory_efficient_dataloaders(data_path, user_ids, samples_per_user,
                                        normalization='none', batch_size=16,
                                        augment_train=False, cache_size=100):
    """
    FIXED: Create memory-efficient data loaders with proper handling of pre-normalized data
    """
    print("Creating FIXED memory-efficient data loaders...")

    # Automatically detect if we should skip normalization
    if normalization in ['log_scale', 'decibel', 'min_max', 'z_score']:
        print(f"⚠️  WARNING: Requested '{normalization}' but data appears pre-normalized.")
        print("   Switching to 'none' to avoid double normalization.")
        normalization = 'none'

    # Get all file paths and labels
    file_paths, labels, sessions = get_file_paths_and_labels(data_path, user_ids)

    if len(file_paths) == 0:
        raise ValueError("No valid spectrogram files found")

    print(f"Total files found: {len(file_paths)}")

    # Convert to numpy arrays for easier manipulation
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    # Limit samples per user
    if samples_per_user is not None:
        limited_paths = []
        limited_labels = []

        for user_idx in np.unique(labels):
            user_mask = labels == user_idx
            user_paths = file_paths[user_mask]
            user_labels = labels[user_mask]

            # Randomly sample if we have more than requested
            if len(user_paths) > samples_per_user:
                indices = np.random.choice(len(user_paths), samples_per_user, replace=False)
                user_paths = user_paths[indices]
                user_labels = user_labels[indices]

            limited_paths.extend(user_paths)
            limited_labels.extend(user_labels)

        file_paths = np.array(limited_paths)
        labels = np.array(limited_labels)

    print(f"Using {len(file_paths)} files after limiting samples per user")

    # FIXED: Better data splitting logic
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for user_idx in np.unique(labels):
        user_mask = labels == user_idx
        user_paths = file_paths[user_mask]
        user_labels = labels[user_mask]

        n_samples = len(user_paths)

        if n_samples < 3:
            # If too few samples, put all in training
            train_paths.extend(user_paths)
            train_labels.extend(user_labels)
            continue

        # FIXED splitting logic
        n_train = max(int(n_samples * 0.7), 1)
        n_val = max(int(n_samples * 0.15), 1)
        n_test = n_samples - n_train - n_val

        # Ensure we have at least 1 test sample
        if n_test < 1:
            n_test = 1
            if n_samples >= 3:
                n_val = max(1, n_samples - n_train - n_test)
            else:
                n_val = 0

        # Random permutation for splitting
        indices = np.random.permutation(n_samples)

        train_end = n_train
        val_end = n_train + n_val

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end] if n_val > 0 else []
        test_idx = indices[val_end:] if n_test > 0 else []

        train_paths.extend(user_paths[train_idx])
        train_labels.extend(user_labels[train_idx])

        if len(val_idx) > 0:
            val_paths.extend(user_paths[val_idx])
            val_labels.extend(user_labels[val_idx])

        if len(test_idx) > 0:
            test_paths.extend(user_paths[test_idx])
            test_labels.extend(user_labels[test_idx])

    # Verify all classes are in training set
    train_classes = set(train_labels)
    all_classes = set(range(len(user_ids)))
    missing_classes = all_classes - train_classes

    if missing_classes:
        print(f"ERROR: Missing classes in training: {missing_classes}")
        # Emergency fix: move samples from val/test to train
        for missing_class in missing_classes:
            # Find this class in val or test
            val_mask = np.array(val_labels) == missing_class
            test_mask = np.array(test_labels) == missing_class

            if np.any(val_mask):
                idx = np.where(val_mask)[0][0]
                train_paths.append(val_paths[idx])
                train_labels.append(val_labels[idx])
                val_paths.pop(idx)
                val_labels.pop(idx)
                print(f"  Moved class {missing_class} sample from val to train")
            elif np.any(test_mask):
                idx = np.where(test_mask)[0][0]
                train_paths.append(test_paths[idx])
                train_labels.append(test_labels[idx])
                test_paths.pop(idx)
                test_labels.pop(idx)
                print(f"  Moved class {missing_class} sample from test to train")

    # Create datasets with NO normalization (data is already normalized)
    train_dataset = SpectrogramDataset(train_paths, train_labels, 'none',
                                       add_channel_dim=True, augment=augment_train,
                                       cache_size=cache_size)
    val_dataset = SpectrogramDataset(val_paths, val_labels, 'none',
                                     add_channel_dim=True, augment=False,
                                     cache_size=cache_size // 2)
    test_dataset = SpectrogramDataset(test_paths, test_labels, 'none',
                                      add_channel_dim=True, augment=False,
                                      cache_size=cache_size // 2)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, persistent_workers=False)

    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Final validation
    print("Final validation of data loaders...")
    train_batch = next(iter(train_loader))
    X_sample, y_sample = train_batch
    print(f"Sample batch - Shape: {X_sample.shape}, Range: [{X_sample.min():.3f}, {X_sample.max():.3f}]")
    print(f"Labels in sample: {torch.unique(y_sample).tolist()}")

    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.tolist())
    print(sorted(set(all_labels)))
    print("Num unique labels:", len(set(all_labels)))

    return train_loader, val_loader, test_loader, sessions


# Quick test function to verify the fix
def test_fixed_loader(data_path, user_ids, samples_per_user=50):
    """
    Test the fixed data loader
    """
    print("Testing FIXED data loader...")

    # Create fixed data loaders
    train_loader, val_loader, test_loader, _ = create_memory_efficient_dataloaders(
        data_path=data_path,
        user_ids=user_ids,
        samples_per_user=samples_per_user,
        normalization='none',  # CRITICAL: No normalization on pre-normalized data
        batch_size=16,
        augment_train=False,
        cache_size=50
    )

    # Test a simple model
    sample_batch = next(iter(train_loader))
    X_batch, y_batch = sample_batch

    print(f"Fixed loader output:")
    print(f"  Shape: {X_batch.shape}")
    print(f"  Range: [{X_batch.min().item():.4f}, {X_batch.max().item():.4f}]")
    print(f"  Mean: {X_batch.mean().item():.4f}")
    print(f"  Std: {X_batch.std().item():.4f}")

    # Quick learning test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_shape, num_classes):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            input_size = np.prod(input_shape)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_size, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(self.flatten(x))

    model = SimpleModel(X_batch.shape[1:], len(user_ids)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Testing learning with fixed data on {device}...")

    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)

            # Check for gradient explosion
            if loss.item() > 1000 or not np.isfinite(loss.item()):
                print(f"  ERROR: Loss explosion at epoch {epoch + 1}, batch {batch_idx}: {loss.item()}")
                break

            loss.backward()

            # Add gradient clipping as safety
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            if batch_idx >= 20:  # Limit for testing
                break

        accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = epoch_loss / (batch_idx + 1)

        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    print(f"Final accuracy: {accuracy:.2f}% (Random baseline: {100 / len(user_ids):.2f}%)")

    if accuracy > (100 / len(user_ids)) * 2:
        print("✅ Fixed! Model is now learning properly!")
        return True
    else:
        print("❌ Still having issues...")
        return False


# Usage
if __name__ == "__main__":
    # Test with your data
    DATA_PATH = "/app/data/grouped_embeddings"
    USER_IDS = list(range(1, 110))  # Increased to more users for better evaluation

    test_fixed_loader(DATA_PATH, USER_IDS)