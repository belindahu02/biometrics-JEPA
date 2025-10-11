import pandas as pd
import numpy as np
import os
import mne
import gc


def standardize(data):
    """Manual standardization to replace sklearn StandardScaler"""
    mean = np.mean(data, axis=0, dtype=np.float32)
    std = np.std(data, axis=0, dtype=np.float32)
    std[std == 0] = 1  # Avoid division by zero
    return (data - mean) / std, mean, std


def apply_standardization(data, mean, std):
    """Apply pre-computed standardization"""
    return (data - mean) / std


def load_edf_file(filepath, max_channels=64):
    """Load a single EDF file efficiently"""
    try:
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

        # Get EEG channels only (exclude EOG, etc.)
        eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EOG')]
        raw.pick_channels(eeg_channels[:max_channels])

        # Load data in chunks to save memory
        raw.load_data()
        data = raw.get_data().T.astype(np.float32)  # Use float32 instead of float64

        # Clear the raw object
        del raw
        gc.collect()

        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_windows(data, frame_size=30, overlap=0.5):
    """Create sliding windows efficiently using stride tricks"""
    if data.shape[0] < frame_size:
        return None

    stride = int(frame_size * (1 - overlap))
    n_windows = (data.shape[0] - frame_size) // stride + 1

    # Use stride tricks for memory efficiency
    shape = (n_windows, frame_size, data.shape[1])
    strides = (stride * data.strides[0], data.strides[0], data.strides[1])

    windowed = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # Copy to avoid memory issues with views
    return windowed.copy()


def data_load_origin(path, users, folders, frame_size=30, max_samples_per_user=None):
    """
    Load EEG data from EDF files with session-level splitting
    Memory efficient version with optional sample limiting
    """
    sessions = []

    # Define session splits (out of 14 runs per user)
    train_runs = list(range(1, 11))  # R01-R10
    val_runs = [11, 12]  # R11-R12
    test_runs = [13, 14]  # R13-R14

    if 'TrainingSet' in folders:
        use_runs = train_runs
    elif 'TestingSet' in folders:
        use_runs = val_runs
    elif 'TestingSet_secret' in folders:
        use_runs = test_runs
    else:
        use_runs = train_runs

    # Use memory-mapped files for large datasets
    all_data = []
    all_labels = []

    for user_id, user in enumerate(users):
        count = 0
        user_data = []
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"Warning: User folder {user_path} not found")
            continue

        for run in use_runs:
            filename = f"S{user:03d}R{run:02d}.edf"
            filepath = os.path.join(user_path, filename)

            data = load_edf_file(filepath)
            if data is None:
                continue

            # Create windows
            windowed = create_windows(data, frame_size=frame_size, overlap=0.5)

            # Clear original data
            del data
            gc.collect()

            if windowed is not None and windowed.shape[0] > 0:
                user_data.append(windowed)
                count += 1

        if len(user_data) > 0:
            # Concatenate all runs for this user
            user_data = np.concatenate(user_data, axis=0)

            # Optional: limit samples per user to save memory
            if max_samples_per_user is not None and user_data.shape[0] > max_samples_per_user:
                indices = np.random.choice(user_data.shape[0], max_samples_per_user, replace=False)
                user_data = user_data[indices]

            all_data.append(user_data)
            all_labels.extend([user_id] * user_data.shape[0])

            print(f"User {user:03d}: {user_data.shape[0]} samples from {count} sessions")

        sessions.append(count)

        # Clear memory after each user
        del user_data
        gc.collect()

    if len(all_data) == 0:
        return np.array([]).reshape(0, frame_size, 64), np.array([]), sessions

    # Concatenate all users
    x_train = np.concatenate(all_data, axis=0)
    y_train = np.array(all_labels)

    # Clear temporary lists
    del all_data, all_labels
    gc.collect()

    print(f"Total samples: {x_train.shape[0]}")

    return x_train, y_train, sessions


def norma_origin(x_all):
    """Normalize data using manual standardization"""
    if x_all.shape[0] == 0:
        return x_all

    original_shape = x_all.shape
    x = x_all.reshape(-1, x_all.shape[-1])
    x, _, _ = standardize(x)
    x_all = x.reshape(original_shape)

    del x
    gc.collect()

    return x_all


def user_data_split(x, y, samples_per_user):
    """Split data to use only specified samples per user"""
    users, counts = np.unique(y, return_counts=True)
    x_train = []
    y_train = []

    for user in users:
        indx = np.where(y == user)[0]
        np.random.shuffle(indx)
        indx = indx[:samples_per_user]
        x_train.append(x[indx])
        y_train.extend(y[indx])

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)

    return x_train, y_train


def norma(x_train, x_val, x_test):
    """Normalize train/val/test with manual standardization"""
    if x_train.shape[0] == 0:
        return x_train, x_val, x_test

    # Fit on training data
    original_shape_train = x_train.shape
    x = x_train.reshape(-1, x_train.shape[-1])
    x, mean, std = standardize(x)
    x_train = x.reshape(original_shape_train)

    del x
    gc.collect()

    # Transform validation data
    if x_val.shape[0] > 0:
        original_shape_val = x_val.shape
        x = x_val.reshape(-1, x_val.shape[-1])
        x = apply_standardization(x, mean, std)
        x_val = x.reshape(original_shape_val)

        del x
        gc.collect()

    # Transform test data
    if x_test.shape[0] > 0:
        original_shape_test = x_test.shape
        x = x_test.reshape(-1, x_test.shape[-1])
        x = apply_standardization(x, mean, std)
        x_test = x.reshape(original_shape_test)

        del x
        gc.collect()

    return x_train, x_val, x_test


def norma_pre(x_all):
    """Normalize pre-training data"""
    if x_all.shape[0] == 0:
        return x_all

    original_shape = x_all.shape
    x = x_all.reshape(-1, x_all.shape[-1])
    x, _, _ = standardize(x)
    x_all = x.reshape(original_shape)

    del x
    gc.collect()

    return x_all