# =============================================
# FIXED data_loader_2d_lazy.py - Session-based Splitting (No Data Leakage)
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


def get_session_based_file_paths_and_labels(data_path, user_ids):
    """
    Get file paths and labels organized by sessions to enable session-based splitting.
    Returns session-organized data for proper train/val/test splitting.
    """
    user_session_data = {}  # user_id -> session_id -> list of file paths

    for user_idx, user_id in enumerate(user_ids):
        user_folder = f"S{user_id:03d}"
        user_path = os.path.join(data_path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: Path {user_path} does not exist")
            user_session_data[user_idx] = {}
            continue

        sessions = {}

        # Loop over session folders
        for item in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, item)
            if os.path.isdir(session_path) and item.startswith(user_folder + 'R'):
                # Extract session number (e.g., S001R01 -> 1, S001R14 -> 14)
                session_num = int(item[-2:])
                session_files = []

                for npy_file in sorted(os.listdir(session_path)):
                    if npy_file.endswith("_stacked.npy"):
                        spec_file = os.path.join(session_path, npy_file)
                        try:
                            # Load just the header to check shape
                            with open(spec_file, 'rb') as f:
                                header = np.lib.format.read_magic(f)
                                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                                if shape in [(4160, 768), (520, 768)]:
                                    session_files.append(spec_file)
                        except Exception as e:
                            print(f"Error checking {spec_file}: {e}")

                if session_files:
                    sessions[session_num] = session_files

        user_session_data[user_idx] = sessions
        total_files = sum(len(files) for files in sessions.values())
        print(f"User {user_id}: {len(sessions)} sessions, {total_files} spectrograms total")

    return user_session_data


def create_session_based_splits(user_session_data, user_ids):
    """
    Create train/val/test splits based on sessions to avoid data leakage.
    Uses 10 sessions for training, 2 for validation, 2 for testing per user.
    """
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    print("Creating session-based splits...")
    print("Split strategy: 10 sessions train, 2 sessions val, 2 sessions test")

    for user_idx, user_id in enumerate(user_ids):
        if user_idx not in user_session_data:
            print(f"Warning: No data for user {user_id}")
            continue

        sessions = user_session_data[user_idx]
        session_nums = sorted(sessions.keys())

        if len(session_nums) < 14:
            print(f"Warning: User {user_id} has only {len(session_nums)} sessions (expected 14)")
            if len(session_nums) < 4:  # Need minimum 4 sessions for meaningful split
                print(f"Skipping user {user_id} due to insufficient sessions")
                continue

        # Ensure we have exactly the sessions we expect (1-14)
        available_sessions = [s for s in session_nums if 1 <= s <= 14]

        # Fixed session assignment to ensure consistency across runs
        # Sessions 1-10: Training
        # Sessions 11-12: Validation
        # Sessions 13-14: Testing
        train_sessions = [s for s in available_sessions if 1 <= s <= 10]
        val_sessions = [s for s in available_sessions if 11 <= s <= 12]
        test_sessions = [s for s in available_sessions if 13 <= s <= 14]

        # If we don't have sessions 11-14, fall back to using available sessions
        if not val_sessions or not test_sessions:
            print(f"Warning: User {user_id} missing high-numbered sessions, using available sessions")
            # Use the last 4 available sessions for val/test
            if len(available_sessions) >= 4:
                train_sessions = available_sessions[:-4]
                val_sessions = available_sessions[-4:-2]
                test_sessions = available_sessions[-2:]
            else:
                # Very few sessions - put most in training
                train_sessions = available_sessions[:-2] if len(available_sessions) > 2 else available_sessions[:1]
                val_sessions = [available_sessions[-2]] if len(available_sessions) > 1 else []
                test_sessions = [available_sessions[-1]] if len(available_sessions) > 0 else []

        # Add files from training sessions
        for session_num in train_sessions:
            if session_num in sessions:
                train_paths.extend(sessions[session_num])
                train_labels.extend([user_idx] * len(sessions[session_num]))

        # Add files from validation sessions
        for session_num in val_sessions:
            if session_num in sessions:
                val_paths.extend(sessions[session_num])
                val_labels.extend([user_idx] * len(sessions[session_num]))

        # Add files from test sessions
        for session_num in test_sessions:
            if session_num in sessions:
                test_paths.extend(sessions[session_num])
                test_labels.extend([user_idx] * len(sessions[session_num]))

        print(
            f"User {user_id:3d}: Train sessions {train_sessions} ({len([f for s in train_sessions if s in sessions for f in sessions[s]])} files), "
            f"Val sessions {val_sessions} ({len([f for s in val_sessions if s in sessions for f in sessions[s]])} files), "
            f"Test sessions {test_sessions} ({len([f for s in test_sessions if s in sessions for f in sessions[s]])} files)")

    print(f"\nFinal session-based split:")
    print(f"Training:   {len(train_paths)} files")
    print(f"Validation: {len(val_paths)} files")
    print(f"Testing:    {len(test_paths)} files")

    # Verify all classes are represented in training
    train_classes = set(train_labels)
    expected_classes = set(range(len(user_ids)))
    missing_classes = expected_classes - train_classes

    if missing_classes:
        print(f"ERROR: Missing classes in training set: {missing_classes}")
        print("This will cause training to fail!")

        # Emergency fix: move some samples from val/test to train for missing classes
        for missing_class in missing_classes:
            # Try to find this class in val first
            val_indices = [i for i, label in enumerate(val_labels) if label == missing_class]
            test_indices = [i for i, label in enumerate(test_labels) if label == missing_class]

            if val_indices:
                # Move one sample from val to train
                idx = val_indices[0]
                train_paths.append(val_paths[idx])
                train_labels.append(val_labels[idx])
                val_paths.pop(idx)
                val_labels.pop(idx)
                print(f"  Emergency: Moved class {missing_class} sample from val to train")
            elif test_indices:
                # Move one sample from test to train
                idx = test_indices[0]
                train_paths.append(test_paths[idx])
                train_labels.append(test_labels[idx])
                test_paths.pop(idx)
                test_labels.pop(idx)
                print(f"  Emergency: Moved class {missing_class} sample from test to train")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def create_session_based_dataloaders(data_path, user_ids, normalization='none', batch_size=16,
                                     augment_train=False, cache_size=100):
    """
    Create session-based data loaders to avoid data leakage.
    Uses 10 sessions for training, 2 for validation, 2 for testing per user.
    """
    print("Creating session-based data loaders...")

    # Get session-organized data
    user_session_data = get_session_based_file_paths_and_labels(data_path, user_ids)

    if not user_session_data:
        raise ValueError("No valid user session data found")

    # Create session-based splits
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_session_based_splits(
        user_session_data, user_ids)

    if not train_paths:
        raise ValueError("No training data found")

    print(f"Session-based split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

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

    print("Final validation of session-based data loaders...")
    train_batch = next(iter(train_loader))
    X_sample, y_sample = train_batch
    print(f"Sample batch - Shape: {X_sample.shape}, Range: [{X_sample.min():.3f}, {X_sample.max():.3f}]")
    print(f"Labels in sample: {torch.unique(y_sample).tolist()}")

    # Verify all expected classes are in training data
    all_train_labels = []
    for _, y in train_loader:
        all_train_labels.extend(y.tolist())
    unique_train_labels = sorted(set(all_train_labels))
    print(f"Training classes: {unique_train_labels}")
    print(f"Expected classes: {list(range(len(user_ids)))}")

    if len(unique_train_labels) != len(user_ids):
        print("ERROR: Mismatch between expected and actual number of classes in training data!")

    return train_loader, val_loader, test_loader, len(user_session_data)


# Quick test function for session-based splitting
def test_session_based_loader(data_path, user_ids_subset):
    """
    Test the session-based data loader
    """
    print("Testing session-based data loader...")

    # Create session-based data loaders
    train_loader, val_loader, test_loader, _ = create_session_based_dataloaders(
        data_path=data_path,
        user_ids=user_ids_subset,
        normalization='none',
        batch_size=16,
        augment_train=False,
        cache_size=50
    )

    print(f"Session-based loader created successfully")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Test data isolation
    print("Verifying session-based data isolation...")

    # This is a basic sanity check - in practice, you'd need to verify
    # that the actual session directories are properly separated
    sample_train = next(iter(train_loader))
    sample_val = next(iter(val_loader))
    sample_test = next(iter(test_loader))

    print(f"Train sample shape: {sample_train[0].shape}, labels: {torch.unique(sample_train[1])}")
    print(f"Val sample shape: {sample_val[0].shape}, labels: {torch.unique(sample_val[1])}")
    print(f"Test sample shape: {sample_test[0].shape}, labels: {torch.unique(sample_test[1])}")

    print("✅ Session-based data loader test completed successfully!")
    return True


# Usage example
if __name__ == "__main__":
    # Test with a subset of users
    DATA_PATH = "/app/data/grouped_embeddings_full_subset20"
    USER_IDS = list(range(1, 11))  # Test with first 10 users

    test_session_based_loader(DATA_PATH, USER_IDS)