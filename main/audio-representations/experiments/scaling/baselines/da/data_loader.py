import numpy as np
import os
import mne
import gc


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
        windows = np.empty((n_windows, frame_size, n_channels), dtype=data.dtype)

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


def data_load_eeg_mmi(path, users, frame_size=40):
    """
    Load EEG MMI dataset with session-based splitting to avoid data leakage.
    Memory-efficient version with incremental loading and dtype optimization.

    Args:
        path: Base path to EEG MMI dataset
        users: List of user IDs (e.g., [1, 2, 3, ...])
        frame_size: Size of sliding window frames

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, sessions
        Each x is shape (n_samples, frame_size, n_channels)
    """
    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []
    sessions = []

    # Session allocation (no overlap to prevent data leakage)
    train_sessions = list(range(1, 11))  # R01-R10 for training
    val_sessions = [11, 12]  # R11-R12 for validation
    test_sessions = [13, 14]  # R13-R14 for testing

    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: User folder {user_folder} not found")
            sessions.append(0)
            continue

        count = 0

        # Load training sessions
        for session in train_sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            if os.path.exists(filepath):
                data = load_edf_session(filepath, frame_size)
                if data is not None:
                    # Convert to float32 to save memory (default is float64)
                    data = data.astype(np.float32)
                    x_train_list.append(data)
                    y_train_list.extend([user_id] * data.shape[0])
                    count += 1
                    del data
                    gc.collect()

        # Load validation sessions
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

        # Load testing sessions
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

        # Periodic garbage collection
        if user_id % 10 == 0:
            gc.collect()

    # Concatenate all data
    print("Concatenating training data...")
    x_train = np.concatenate(x_train_list, axis=0).astype(np.float32) if x_train_list else np.array([])
    y_train = np.array(y_train_list, dtype=np.int32)

    # Clear training lists to free memory
    del x_train_list, y_train_list
    gc.collect()

    print("Concatenating validation data...")
    x_val = np.concatenate(x_val_list, axis=0).astype(np.float32) if x_val_list else np.array([])
    y_val = np.array(y_val_list, dtype=np.int32)

    # Clear validation lists
    del x_val_list, y_val_list
    gc.collect()

    print("Concatenating test data...")
    x_test = np.concatenate(x_test_list, axis=0).astype(np.float32) if x_test_list else np.array([])
    y_test = np.array(y_test_list, dtype=np.int32)

    # Clear test lists
    del x_test_list, y_test_list
    gc.collect()

    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test, sessions


def norma(x_train, x_val, x_test):
    """
    Normalize data using training set statistics.
    Fits on training data and applies to all splits.
    Memory-efficient version using float32.
    """
    # Ensure float32 dtype
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Reshape for normalization
    x_train_flat = np.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))

    # Calculate statistics from training data
    mean = np.mean(x_train_flat, axis=0, dtype=np.float32)
    std = np.std(x_train_flat, axis=0, dtype=np.float32)
    std[std == 0] = 1  # Avoid division by zero

    # Normalize training data
    x_train_normalized = (x_train_flat - mean) / std
    x_train = np.reshape(x_train_normalized, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Clear intermediate arrays
    del x_train_flat, x_train_normalized
    gc.collect()

    # Normalize validation data
    x_val_flat = np.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
    x_val_normalized = (x_val_flat - mean) / std
    x_val = np.reshape(x_val_normalized, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))

    del x_val_flat, x_val_normalized
    gc.collect()

    # Normalize test data
    x_test_flat = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2]))
    x_test_normalized = (x_test_flat - mean) / std
    x_test = np.reshape(x_test_normalized, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    del x_test_flat, x_test_normalized
    gc.collect()

    return x_train, x_val, x_test