import pandas as pd
import numpy as np
import os
from transformations import *
import mne
import gc


def load_edf_file(filepath, max_samples=None):
    """
    Load EDF file and extract EEG data with optional downsampling

    Args:
        filepath: Path to EDF file
        max_samples: If set, downsample to this many samples max
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        # Get EEG channel data
        data = raw.get_data().T  # Transpose to (samples, channels)

        # Optional: Downsample if data is too large
        if max_samples is not None and data.shape[0] > max_samples:
            # Simple downsampling by taking every nth sample
            step = data.shape[0] // max_samples
            data = data[::step]

        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_normalization_stats(path, users, sessions, frame_size=30, max_samples_per_session=None):
    """
    Compute mean and std statistics from training data for normalization.
    This does a single pass through the data to compute statistics.

    Returns:
        mean, std: Arrays of shape (n_channels,)
    """
    print("Computing normalization statistics...")

    # Running statistics
    n_samples = 0
    sum_x = None
    sum_x2 = None

    for user in users:
        user_folder = f"S{user:03d}"

        for session in sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(path, user_folder, filename)

            try:
                data = load_edf_file(filepath, max_samples=max_samples_per_session)
                if data is None or data.shape[0] < frame_size:
                    continue

                # Truncate to multiple of frame_size
                data = data[:(data.shape[0] // frame_size) * frame_size]

                if data.shape[0] == 0:
                    continue

                # Initialize arrays on first valid data
                if sum_x is None:
                    sum_x = np.zeros(data.shape[1])
                    sum_x2 = np.zeros(data.shape[1])

                # Update running statistics
                sum_x += np.sum(data, axis=0)
                sum_x2 += np.sum(data ** 2, axis=0)
                n_samples += data.shape[0]

                del data
                gc.collect()

            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

    # Compute mean and std
    mean = sum_x / n_samples
    std = np.sqrt(sum_x2 / n_samples - mean ** 2)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    print(f"Computed stats from {n_samples} samples")
    return mean, std


class EEGDataGenerator:
    """
    Generator for streaming EEG data in batches without loading everything into memory
    """

    def __init__(self, path, users, sessions, frame_size=30, batch_size=32,
                 max_samples_per_session=None, mean=None, std=None, shuffle=True):
        """
        Args:
            path: Base path to dataset
            users: List of user IDs
            sessions: List of session numbers
            frame_size: Window size for sliding windows
            batch_size: Number of samples per batch
            max_samples_per_session: Limit samples per session
            mean, std: Normalization statistics (if None, data won't be normalized)
            shuffle: Whether to shuffle data each epoch
        """
        self.path = path
        self.users = users
        self.sessions = sessions
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.max_samples_per_session = max_samples_per_session
        self.mean = mean
        self.std = std
        self.shuffle = shuffle

        # Build file list
        self.file_list = []
        for user_id, user in enumerate(users):
            user_folder = f"S{user:03d}"
            for session in sessions:
                filename = f"S{user:03d}R{session:02d}.edf"
                filepath = os.path.join(path, user_folder, filename)
                self.file_list.append((filepath, user_id))

        self.n_files = len(self.file_list)
        print(f"Generator initialized with {self.n_files} files")

    def _process_file(self, filepath, user_id):
        """Process a single file and return windows"""
        data = load_edf_file(filepath, max_samples=self.max_samples_per_session)
        if data is None or data.shape[0] < self.frame_size:
            return None, None

        # Truncate to multiple of frame_size
        data = data[:(data.shape[0] // self.frame_size) * self.frame_size]

        if data.shape[0] == 0:
            return None, None

        # Create sliding windows with 50% overlap
        windows = np.lib.stride_tricks.sliding_window_view(
            data, (self.frame_size, data.shape[1])
        )[::self.frame_size // 2, :]

        # Reshape from (n_windows, 1, frame_size, n_channels) to (n_windows, frame_size, n_channels)
        if len(windows.shape) == 4:
            windows = windows.squeeze(axis=1)
        elif len(windows.shape) != 3:
            print(f"Warning: Unexpected shape {windows.shape}")
            return None, None

        # Normalize if statistics are provided
        if self.mean is not None and self.std is not None:
            # Reshape, normalize, reshape back
            original_shape = windows.shape
            windows = windows.reshape(-1, windows.shape[-1])
            windows = (windows - self.mean) / self.std
            windows = windows.reshape(original_shape)

        labels = np.full(windows.shape[0], user_id, dtype=np.int32)

        return windows, labels

    def __iter__(self):
        """Iterator that yields batches of data - FIXED to work with multiple epochs"""
        while True:  # CRITICAL FIX: Infinite loop to support multiple epochs
            # Shuffle file list if requested
            file_indices = np.arange(self.n_files)
            if self.shuffle:
                np.random.shuffle(file_indices)

            batch_x = []
            batch_y = []

            for idx in file_indices:
                filepath, user_id = self.file_list[idx]

                try:
                    windows, labels = self._process_file(filepath, user_id)

                    if windows is None:
                        continue

                    # Shuffle windows within file if requested
                    if self.shuffle:
                        perm = np.random.permutation(len(windows))
                        windows = windows[perm]
                        labels = labels[perm]

                    # Add to batch
                    for i in range(len(windows)):
                        batch_x.append(windows[i])
                        batch_y.append(labels[i])

                        if len(batch_x) == self.batch_size:
                            yield np.array(batch_x), np.array(batch_y)
                            batch_x = []
                            batch_y = []

                    del windows, labels
                    gc.collect()

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue

            # Yield remaining samples at end of epoch
            if len(batch_x) > 0:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

            # After going through all files once, loop continues for next epoch

    def get_steps_per_epoch(self):
        """Estimate number of batches per epoch"""
        # This is an estimate; actual value may vary slightly
        total_samples = 0
        for filepath, _ in self.file_list:
            try:
                data = load_edf_file(filepath, max_samples=self.max_samples_per_session)
                if data is not None and data.shape[0] >= self.frame_size:
                    n_samples = (data.shape[0] // self.frame_size) * self.frame_size
                    n_windows = len(range(0, n_samples - self.frame_size + 1, self.frame_size // 2))
                    total_samples += n_windows
                del data
                gc.collect()
            except:
                continue

        return max(1, total_samples // self.batch_size)


def data_load_with_generators(path, users, frame_size=30, batch_size=32,
                              max_samples_per_session=None):
    """
    Create data generators for train/val/test splits with proper normalization.

    Args:
        path: Base path to dataset
        users: List of user IDs
        frame_size: Window size for sliding windows
        batch_size: Batch size for generators
        max_samples_per_session: Limit samples per session

    Returns:
        train_gen, val_gen, test_gen, steps_per_epoch dict
    """
    train_sessions = list(range(1, 11))  # R01-R10
    val_sessions = [11, 12]  # R11-R12
    test_sessions = [13, 14]  # R13-R14

    # Compute normalization statistics from training data only
    print("Computing normalization statistics from training data...")
    mean, std = compute_normalization_stats(
        path, users, train_sessions, frame_size, max_samples_per_session
    )

    # Create generators
    print("Creating training generator...")
    train_gen = EEGDataGenerator(
        path, users, train_sessions, frame_size, batch_size,
        max_samples_per_session, mean, std, shuffle=True
    )

    print("Creating validation generator...")
    val_gen = EEGDataGenerator(
        path, users, val_sessions, frame_size, batch_size,
        max_samples_per_session, mean, std, shuffle=False
    )

    print("Creating test generator...")
    test_gen = EEGDataGenerator(
        path, users, test_sessions, frame_size, batch_size,
        max_samples_per_session, mean, std, shuffle=False
    )

    # Get steps per epoch (optional, for progress tracking)
    steps = {
        'train': train_gen.get_steps_per_epoch(),
        'val': val_gen.get_steps_per_epoch(),
        'test': test_gen.get_steps_per_epoch()
    }

    print(f"Estimated steps per epoch - Train: {steps['train']}, Val: {steps['val']}, Test: {steps['test']}")

    return train_gen, val_gen, test_gen, steps


class AugmentedEEGDataGenerator:
    """
    Generator that applies data augmentation on-the-fly
    """

    def __init__(self, base_generator, transformations, sigma_l, ext=False):
        """
        Args:
            base_generator: Base EEGDataGenerator instance
            transformations: List of transformation functions
            sigma_l: List of sigma values for transformations
            ext: Whether to use extended augmentation
        """
        self.base_generator = base_generator
        self.transformations = transformations
        self.sigma_l = sigma_l
        self.ext = ext
        self.n_transforms = len(transformations)

    def __iter__(self):
        """Yields augmented batches"""
        for batch_x, batch_y in self.base_generator:
            # For each transformation
            for i, (transform, sigma) in enumerate(zip(self.transformations, self.sigma_l)):
                # Original samples (negative examples)
                yield batch_x, np.zeros(len(batch_x), dtype=bool)

                # Augmented samples (positive examples)
                augmented = np.array([transform(x, sigma=sigma) for x in batch_x])
                yield augmented, np.ones(len(batch_x), dtype=bool)

                # Extended augmentation: other transformations as negative examples
                if self.ext:
                    for j, (other_transform, other_sigma) in enumerate(zip(self.transformations, self.sigma_l)):
                        if i != j:
                            other_augmented = np.array([other_transform(x, sigma=other_sigma) for x in batch_x])
                            yield other_augmented, np.zeros(len(batch_x), dtype=bool)
