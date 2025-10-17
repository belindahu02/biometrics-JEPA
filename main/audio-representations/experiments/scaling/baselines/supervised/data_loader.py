import pandas as pd
import numpy as np
import os
import mne
import gc
from tensorflow.keras.utils import Sequence
from collections import OrderedDict


class SessionCache:
    """
    LRU cache for EEG sessions to balance memory and speed.
    Keeps recently used sessions in memory.
    """

    def __init__(self, max_cache_size_gb=8):
        """
        Args:
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size_gb * 1024 ** 3  # Convert to bytes
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def _estimate_size(self, data):
        """Estimate memory size of numpy array in bytes"""
        return data.nbytes

    def get(self, filepath):
        """Get session data from cache or None if not cached"""
        if filepath in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(filepath)
            self.hits += 1
            return self.cache[filepath]
        self.misses += 1
        return None

    def put(self, filepath, data):
        """Add session data to cache, evicting old entries if needed"""
        data_size = self._estimate_size(data)

        # Evict old entries if needed
        while self.current_size + data_size > self.max_cache_size and len(self.cache) > 0:
            oldest_key, oldest_data = self.cache.popitem(last=False)
            self.current_size -= self._estimate_size(oldest_data)
            del oldest_data

        # Add new entry
        self.cache[filepath] = data
        self.current_size += data_size

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.current_size = 0
        gc.collect()

    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size_mb': self.current_size / (1024 ** 2),
            'num_cached': len(self.cache)
        }


class StreamingEEGDataGenerator(Sequence):
    """
    Memory-efficient data generator with session-level caching.
    Balances memory usage and speed by caching recently used sessions.
    """

    def __init__(self, path, users, split='train', frame_size=30, batch_size=8,
                 shuffle=True, normalization_stats=None, cache_size_gb=8):
        """
        Args:
            path: Base path to dataset
            users: List of user IDs to include
            split: 'train', 'val', or 'test'
            frame_size: Window size for segmentation
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            normalization_stats: Dict with 'mean' and 'std' for normalization
            cache_size_gb: Size of session cache in GB (default 8GB)
        """
        self.path = path
        self.users = users
        self.split = split
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization_stats = normalization_stats

        # Initialize session cache
        self.session_cache = SessionCache(max_cache_size_gb=cache_size_gb)

        # Define session ranges for each split
        if split == 'train':
            self.sessions = list(range(1, 11))  # Sessions 1-10
        elif split == 'val':
            self.sessions = list(range(11, 13))  # Sessions 11-12
        elif split == 'test':
            self.sessions = list(range(13, 15))  # Sessions 13-14
        else:
            raise ValueError(f"Invalid split: {split}")

        # Build index of all available samples
        self._build_index()

        # Initialize sample order
        self.indices = np.arange(len(self.sample_index))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _build_index(self):
        """
        Build an index of all samples without loading the actual data.
        Each entry contains: (user_id, filepath, window_idx)
        """
        self.sample_index = []

        print(f"Building index for {self.split} split...")

        for user_id, user in enumerate(self.users):
            user_folder = f"S{user:03d}"
            user_path = os.path.join(self.path, user_folder)

            if not os.path.exists(user_path):
                continue

            for session in self.sessions:
                filename = f"S{user:03d}R{session:02d}.edf"
                filepath = os.path.join(user_path, filename)

                if not os.path.exists(filepath):
                    continue

                try:
                    # Quick read to get dimensions without loading full data
                    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
                    n_samples = raw.n_times
                    raw.close()
                    del raw

                    # Calculate number of windows
                    if n_samples < self.frame_size:
                        continue

                    n_samples_truncated = (n_samples // self.frame_size) * self.frame_size
                    # 50% overlap
                    n_windows = (n_samples_truncated - self.frame_size) // (self.frame_size // 2) + 1

                    # Add each window to the index
                    for window_idx in range(n_windows):
                        self.sample_index.append({
                            'user_id': user_id,
                            'filepath': filepath,
                            'window_idx': window_idx
                        })

                except Exception as e:
                    continue

        print(f"{self.split} split: {len(self.sample_index)} samples indexed")

        if len(self.sample_index) == 0:
            raise ValueError(f"No samples found for {self.split} split!")

    def _load_session(self, filepath):
        """
        Load a session from cache or disk.
        """
        # Check cache first
        cached_data = self.session_cache.get(filepath)
        if cached_data is not None:
            return cached_data

        # Load from disk
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data = raw.get_data().T.astype(np.float32)  # Shape: (n_samples, n_channels)
        raw.close()
        del raw

        # Truncate to multiple of frame_size
        data = data[:(data.shape[0] // self.frame_size) * self.frame_size]

        # Add to cache
        self.session_cache.put(filepath, data)

        return data

    def _extract_window(self, data, window_idx):
        """Extract a window from session data"""
        stride = self.frame_size // 2
        start_idx = window_idx * stride
        end_idx = start_idx + self.frame_size

        window = data[start_idx:end_idx, :].copy()

        # Apply normalization
        if self.normalization_stats is not None:
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']
            window = (window - mean) / std

        return window

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.sample_index) / self.batch_size))

    def __getitem__(self, idx):
        """
        Generate one batch of data using cached sessions.
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Group samples by file to maximize cache efficiency
        file_groups = OrderedDict()
        for i in batch_indices:
            sample_info = self.sample_index[i]
            filepath = sample_info['filepath']
            if filepath not in file_groups:
                file_groups[filepath] = []
            file_groups[filepath].append(sample_info)

        # Load windows from cached or disk sessions
        batch_X = []
        batch_y = []

        for filepath, samples in file_groups.items():
            # Load session (from cache or disk)
            session_data = self._load_session(filepath)

            # Extract windows
            for sample_info in samples:
                window = self._extract_window(session_data, sample_info['window_idx'])
                batch_X.append(window)
                batch_y.append(sample_info['user_id'])

        return np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.int32)

    def on_epoch_end(self):
        """Shuffle indices after each epoch and print cache stats"""
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Print cache statistics
        stats = self.session_cache.get_stats()
        print(f"\nCache stats - Hit rate: {stats['hit_rate']:.2%}, "
              f"Size: {stats['cache_size_mb']:.1f}MB, "
              f"Cached sessions: {stats['num_cached']}")

    def get_num_classes(self):
        """Get number of unique users"""
        return len(self.users)


def calculate_normalization_stats(path, users, frame_size=30, max_samples=10000):
    """
    Calculate mean and std from training data without loading everything into memory.
    Uses random sampling from training sessions for efficiency.

    Args:
        path: Base path to dataset
        users: List of user IDs
        frame_size: Window size
        max_samples: Maximum number of samples to use for statistics (for speed)

    Returns:
        dict with 'mean' and 'std' arrays
    """
    print("Calculating normalization statistics from training data...")

    train_sessions = list(range(1, 11))  # Sessions 1-10 for training
    samples_collected = 0
    all_samples = []

    for user in users:
        if samples_collected >= max_samples:
            break

        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"{user_path} does not exist")
            continue

        for session in train_sessions:
            if samples_collected >= max_samples:
                break

            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if not os.path.exists(filepath):
                print(f"{filepath} does not exist")
                continue

            try:
                # Load file
                raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
                data = raw.get_data().T.astype(np.float32)
                raw.close()
                del raw

                # Take random samples
                if data.shape[0] >= frame_size:
                    # Randomly sample a few windows from this file
                    data = data[:(data.shape[0] // frame_size) * self.frame_size]
                    num_windows = min(5, data.shape[0] // frame_size)  # Max 5 windows per file

                    for _ in range(num_windows):
                        if samples_collected >= max_samples:
                            break
                        start_idx = np.random.randint(0, data.shape[0] - frame_size + 1)
                        window = data[start_idx:start_idx + frame_size, :]
                        all_samples.append(window.reshape(-1, window.shape[-1]))
                        samples_collected += window.shape[0]

                print(f"All samples: {len(all_samples)}")
                del data
                gc.collect()

            except Exception as e:
                continue

    if len(all_samples) == 0:
        raise ValueError("No samples found for normalization!")

    # Concatenate and calculate statistics
    all_samples = np.concatenate(all_samples, axis=0)
    mean = np.mean(all_samples, axis=0, dtype=np.float32)
    std = np.std(all_samples, axis=0, dtype=np.float32)
    std[std == 0] = 1.0

    del all_samples
    gc.collect()

    print(f"Normalization stats calculated from {samples_collected} samples")

    return {'mean': mean, 'std': std}