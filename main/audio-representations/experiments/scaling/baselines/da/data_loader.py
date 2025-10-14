import numpy as np
import os
import mne
import gc
import tensorflow as tf


def data_load_origin(path, users, folders, frame_size=30):
    """Legacy function - not used with EEG MMI dataset"""
    pass


def norma_origin(x_all):
    """Legacy function - kept for compatibility"""
    x = np.reshape(x_all, (x_all.shape[0] * x_all.shape[1], x_all.shape[2]))
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    x = (x - mean) / std
    x_all = np.reshape(x, (x_all.shape[0], x_all.shape[1], x_all.shape[2]))
    return x_all


def user_data_split(x, y, samples_per_user):
    """Legacy function - kept for compatibility"""
    users, counts = np.unique(y, return_counts=True)
    x_train = np.array([])
    y_train = np.array([])
    for user in users:
        indx = np.where(y == user)[0]
        np.random.shuffle(indx)
        indx = indx[:samples_per_user]
        if x_train.shape[0] == 0:
            x_train = x[indx]
            y_train = y[indx]
        else:
            x_train = np.concatenate((x_train, x[indx]), axis=0)
            y_train = np.concatenate((y_train, y[indx]), axis=0)
    return x_train, y_train


def load_edf_session(filepath, frame_size=40):
    """
    Load a single EDF file and extract sliding windows.
    Returns numpy array of shape (n_windows, frame_size, n_channels)
    Memory-efficient version with explicit cleanup.
    """
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

        # Get data: shape (n_channels, n_timepoints)
        data = raw.get_data()

        # Close and delete raw object to free memory
        raw.close()
        del raw
        gc.collect()

        # Transpose to (n_timepoints, n_channels)
        data = data.T

        # Create sliding windows with 50% overlap
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        stride = frame_size // 2

        # Calculate number of windows
        n_windows = (n_samples - frame_size) // stride + 1

        if n_windows <= 0:
            del data
            gc.collect()
            return None

        # Create sliding windows using numpy's stride tricks (more memory efficient)
        # Pre-allocate the array
        windows = np.empty((n_windows, frame_size, n_channels), dtype=np.float32)

        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + frame_size
            if end_idx <= n_samples:
                windows[i] = data[start_idx:end_idx, :]

        # Clean up intermediate data
        del data
        gc.collect()

        return windows

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        gc.collect()
        return None


def compute_normalization_stats(path, users, frame_size=40):
    """
    Compute normalization statistics from training data without loading all data into memory.
    Only uses training sessions (R01-R10).

    Returns:
        mean, std: Normalization statistics (arrays of shape (n_channels,))
        n_channels: Number of channels in the data
    """
    train_sessions = list(range(1, 11))  # R01-R10 for training

    # First pass: compute mean
    sum_values = None
    total_samples = 0
    n_channels = None

    print("Computing normalization statistics - Pass 1 (mean)...")
    for user in users:
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            continue

        for session in train_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    if n_channels is None:
                        n_channels = data.shape[2]
                        sum_values = np.zeros(n_channels, dtype=np.float64)

                    # Flatten to (n_windows * frame_size, n_channels)
                    flat_data = data.reshape(-1, n_channels)
                    sum_values += np.sum(flat_data, axis=0)
                    total_samples += flat_data.shape[0]

                    del data, flat_data
                    gc.collect()

    mean = (sum_values / total_samples).astype(np.float32)

    # Second pass: compute std
    sum_squared_diff = np.zeros(n_channels, dtype=np.float64)

    print("Computing normalization statistics - Pass 2 (std)...")
    for user in users:
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            continue

        for session in train_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    flat_data = data.reshape(-1, n_channels)
                    sum_squared_diff += np.sum((flat_data - mean) ** 2, axis=0)

                    del data, flat_data
                    gc.collect()

    std = np.sqrt(sum_squared_diff / total_samples).astype(np.float32)
    std[std == 0] = 1  # Avoid division by zero

    print(f"Normalization stats computed: mean shape {mean.shape}, std shape {std.shape}")
    return mean, std, n_channels


def get_file_list_and_labels(path, users, frame_size=40):
    """
    Get list of files and their metadata without loading the actual data.

    Returns:
        train_files, val_files, test_files: Lists of (filepath, user_id, n_windows) tuples
        n_channels: Number of channels
    """
    train_sessions = list(range(1, 11))  # R01-R10
    val_sessions = [11, 12]  # R11-R12
    test_sessions = [13, 14]  # R13-R14

    train_files = []
    val_files = []
    test_files = []
    n_channels = None

    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: User folder {user_folder} not found")
            continue

        # Process training files
        for session in train_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                # Load once to get dimensions
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    if n_channels is None:
                        n_channels = data.shape[2]
                    n_windows = data.shape[0]
                    train_files.append((filepath, user_id, n_windows))
                    del data
                    gc.collect()

        # Process validation files
        for session in val_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    n_windows = data.shape[0]
                    val_files.append((filepath, user_id, n_windows))
                    del data
                    gc.collect()

        # Process test files
        for session in test_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    n_windows = data.shape[0]
                    test_files.append((filepath, user_id, n_windows))
                    del data
                    gc.collect()

    return train_files, val_files, test_files, n_channels


class EEGDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for streaming EEG data.
    Loads data on-the-fly during training to avoid memory issues.
    """

    def __init__(self, file_list, frame_size, mean, std, batch_size=8, shuffle=True):
        """
        Args:
            file_list: List of (filepath, user_id, n_windows) tuples
            frame_size: Size of each window
            mean: Normalization mean
            std: Normalization std
            batch_size: Batch size
            shuffle: Whether to shuffle data after each epoch
        """
        self.file_list = file_list
        self.frame_size = frame_size
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create index mapping: (file_idx, window_idx) for each sample
        self.sample_indices = []
        for file_idx, (filepath, user_id, n_windows) in enumerate(file_list):
            for window_idx in range(n_windows):
                self.sample_indices.append((file_idx, window_idx, user_id))

        self.n_samples = len(self.sample_indices)
        self.indices = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(self.indices)

        # Cache for loaded files to avoid reloading within epoch
        self.file_cache = {}
        self.max_cache_size = 5  # Keep at most 5 files in memory

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, batch_idx):
        """Generate one batch of data"""
        # Get indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Prepare batch arrays
        actual_batch_size = len(batch_indices)
        X_batch = np.empty((actual_batch_size, self.frame_size, len(self.mean)), dtype=np.float32)
        y_batch = np.empty(actual_batch_size, dtype=np.int32)

        # Load data for each sample in batch
        for i, idx in enumerate(batch_indices):
            file_idx, window_idx, user_id = self.sample_indices[idx]
            filepath, _, _ = self.file_list[file_idx]

            # Load file if not in cache
            if file_idx not in self.file_cache:
                # Clear cache if too large
                if len(self.file_cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.file_cache))
                    del self.file_cache[oldest_key]
                    gc.collect()

                # Load file
                data = load_edf_session(filepath, self.frame_size)
                if data is not None:
                    # Normalize
                    data_flat = data.reshape(-1, data.shape[2])
                    data_flat = (data_flat - self.mean) / self.std
                    data = data_flat.reshape(data.shape)
                    self.file_cache[file_idx] = data
                else:
                    # Handle missing data - use zeros
                    self.file_cache[file_idx] = np.zeros((1, self.frame_size, len(self.mean)), dtype=np.float32)

            # Get window from cached file
            X_batch[i] = self.file_cache[file_idx][window_idx]
            y_batch[i] = user_id

        return X_batch, y_batch

    def on_epoch_end(self):
        """Updates after each epoch"""
        # Clear file cache
        self.file_cache.clear()
        gc.collect()

        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_total_samples(self):
        """Return total number of samples"""
        return self.n_samples


def data_load_eeg_mmi_streaming(path, users, frame_size=40):
    """
    Setup streaming data loading for EEG MMI dataset.
    Returns file lists and normalization stats instead of loading all data.

    Returns:
        train_files, val_files, test_files: File lists for generators
        mean, std: Normalization statistics
        n_channels: Number of channels
        train_samples, val_samples, test_samples: Total sample counts
    """
    print("Setting up streaming data loading...")

    # Get file lists
    train_files, val_files, test_files, n_channels = get_file_list_and_labels(
        path, users, frame_size
    )

    print(f"Found {len(train_files)} training files, {len(val_files)} validation files, {len(test_files)} test files")

    # Compute normalization statistics from training data
    mean, std, _ = compute_normalization_stats(path, users, frame_size)

    # Calculate total samples
    train_samples = sum(n_windows for _, _, n_windows in train_files)
    val_samples = sum(n_windows for _, _, n_windows in val_files)
    test_samples = sum(n_windows for _, _, n_windows in test_files)

    print(f"Total samples - Train: {train_samples}, Val: {val_samples}, Test: {test_samples}")

    return train_files, val_files, test_files, mean, std, n_channels, train_samples, val_samples, test_samples


# Keep legacy function for backward compatibility
def data_load_eeg_mmi(path, users, frame_size=40):
    """
    DEPRECATED: This function loads all data into memory and may cause OOM errors.
    Use data_load_eeg_mmi_streaming instead.
    """
    print("WARNING: Using legacy data_load_eeg_mmi - this loads all data into memory!")
    print("Consider using data_load_eeg_mmi_streaming for better memory efficiency.")

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []
    sessions = []

    train_sessions = list(range(1, 11))
    val_sessions = [11, 12]
    test_sessions = [13, 14]

    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: User folder {user_folder} not found")
            sessions.append(0)
            continue

        count = 0

        for session in train_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    data = data.astype(np.float32)
                    x_train_list.append(data)
                    y_train_list.extend([user_id] * data.shape[0])
                    count += 1
                    del data
                    gc.collect()

        for session in val_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    data = data.astype(np.float32)
                    x_val_list.append(data)
                    y_val_list.extend([user_id] * data.shape[0])
                    count += 1
                    del data
                    gc.collect()

        for session in test_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    data = data.astype(np.float32)
                    x_test_list.append(data)
                    y_test_list.extend([user_id] * data.shape[0])
                    count += 1
                    del data
                    gc.collect()

        sessions.append(count)
        print(f"Loaded user {user_folder}: {count} sessions")

        if user_id % 10 == 0:
            gc.collect()

    print("Concatenating training data...")
    x_train = np.concatenate(x_train_list, axis=0).astype(np.float32) if x_train_list else np.array([])
    y_train = np.array(y_train_list, dtype=np.int32)

    del x_train_list, y_train_list
    gc.collect()

    print("Concatenating validation data...")
    x_val = np.concatenate(x_val_list, axis=0).astype(np.float32) if x_val_list else np.array([])
    y_val = np.array(y_val_list, dtype=np.int32)

    del x_val_list, y_val_list
    gc.collect()

    print("Concatenating test data...")
    x_test = np.concatenate(x_test_list, axis=0).astype(np.float32) if x_test_list else np.array([])
    y_test = np.array(y_test_list, dtype=np.int32)

    del x_test_list, y_test_list
    gc.collect()

    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test, sessions


def norma(x_train, x_val, x_test):
    """
    Normalize data using training set statistics.
    """
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train_flat = np.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))

    mean = np.mean(x_train_flat, axis=0, dtype=np.float32)
    std = np.std(x_train_flat, axis=0, dtype=np.float32)
    std[std == 0] = 1

    x_train_normalized = (x_train_flat - mean) / std
    x_train = np.reshape(x_train_normalized, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    del x_train_flat, x_train_normalized
    gc.collect()

    x_val_flat = np.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
    x_val_normalized = (x_val_flat - mean) / std
    x_val = np.reshape(x_val_normalized, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))

    del x_val_flat, x_val_normalized
    gc.collect()

    x_test_flat = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2]))
    x_test_normalized = (x_test_flat - mean) / std
    x_test = np.reshape(x_test_normalized, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    del x_test_flat, x_test_normalized
    gc.collect()

    return x_train, x_val, x_test