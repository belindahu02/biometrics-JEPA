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


def data_load_origin(path, users, sessions, frame_size=30, max_samples_per_session=None):
    """
    Load data from EEG MMI dataset with session-level organization

    Args:
        path: Base path to dataset
        users: List of user IDs
        sessions: List of session numbers
        frame_size: Window size for sliding windows
        max_samples_per_session: Limit samples per session to reduce memory
    """
    x_train = []  # Use list instead of numpy array for memory efficiency
    y_train = []
    session_counts = []

    total_sessions = len(users) * len(sessions)
    loaded_sessions = 0

    for user_id, user in enumerate(users):
        count = 0
        user_folder = f"S{user:03d}"

        for session in sessions:
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(path, user_folder, filename)

            try:
                data = load_edf_file(filepath, max_samples=max_samples_per_session)
                if data is None or data.shape[0] < frame_size:
                    continue

                # Truncate to multiple of frame_size for sliding window
                data = data[:(data.shape[0] // frame_size) * frame_size]

                if data.shape[0] == 0:
                    continue

                # Create sliding windows with 50% overlap
                data = np.lib.stride_tricks.sliding_window_view(
                    data, (frame_size, data.shape[1])
                )[::frame_size // 2, :]

                # Reshape from (n_windows, 1, frame_size, n_channels) to (n_windows, frame_size, n_channels)
                if len(data.shape) == 4:
                    data = data.squeeze(axis=1)
                elif len(data.shape) != 3:
                    print(f"Warning: Unexpected data shape {data.shape} for {filepath}")
                    continue

                # Append to list
                x_train.append(data)
                y_train.extend([user_id] * data.shape[0])

                count += 1
                loaded_sessions += 1

                # Free memory
                del data

                if loaded_sessions % 10 == 0:
                    print(f"Loaded {loaded_sessions}/{total_sessions} sessions...")

            except (FileNotFoundError, IndexError, Exception) as e:
                print(f"Error loading {filepath}: {e}")
                continue

        session_counts.append(count)

        # Force garbage collection after each user
        gc.collect()

    if len(x_train) == 0:
        raise ValueError("No data was loaded! Check your dataset path and file structure.")

    # Convert list to numpy array at the end
    print("Concatenating data arrays...")
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)

    print(f"Loaded data shape: {x_train.shape}")
    return x_train, y_train, session_counts


def norma_origin(x_all):
    """Normalize data using z-score normalization (mean=0, std=1)"""
    if len(x_all.shape) != 3:
        raise ValueError(f"Expected 3D array (samples, timesteps, features), got shape {x_all.shape}")

    x = np.reshape(x_all, (x_all.shape[0] * x_all.shape[1], x_all.shape[2]))

    # Calculate mean and std for each feature
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    # Normalize
    x = (x - mean) / std

    x_all = np.reshape(x, (x_all.shape[0], x_all.shape[1], x_all.shape[2]))

    # Free memory
    del x
    gc.collect()

    return x_all


def user_data_split(x, y, samples_per_user):
    """Split data to use specific number of samples per user"""
    users, counts = np.unique(y, return_counts=True)
    x_train = []
    y_train = []

    for user in users:
        indx = np.where(y == user)[0]
        np.random.shuffle(indx)
        indx = indx[:samples_per_user]
        x_train.append(x[indx])
        y_train.append(y[indx])

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    return x_train, y_train


def data_load(path, users, frame_size=30, max_samples_per_session=None):
    """
    Load EEG MMI data with session-level splitting to prevent data leakage.
    Sessions per user: R01-R14
    - Train: R01-R10 (10 sessions)
    - Val: R11-R12 (2 sessions)
    - Test: R13-R14 (2 sessions)

    Args:
        max_samples_per_session: Limit samples per session (e.g., 10000) to reduce memory
    """
    train_sessions = list(range(1, 11))  # R01-R10
    val_sessions = [11, 12]  # R11-R12
    test_sessions = [13, 14]  # R13-R14

    print("Loading training data...")
    x_train, y_train, sessions_train = data_load_origin(
        path, users, train_sessions, frame_size, max_samples_per_session
    )
    print(f"Training samples: {x_train.shape[0]}")

    print("Loading validation data...")
    x_val, y_val, sessions_val = data_load_origin(
        path, users, val_sessions, frame_size, max_samples_per_session
    )
    print(f"Validation samples: {x_val.shape[0]}")

    print("Loading test data...")
    x_test, y_test, sessions_test = data_load_origin(
        path, users, test_sessions, frame_size, max_samples_per_session
    )
    print(f"Test samples: {x_test.shape[0]}")

    return x_train, y_train, x_val, y_val, x_test, y_test, sessions_train


def norma(x_train, x_val, x_test):
    """Normalize train/val/test data using z-score normalization fit on training data"""
    # Validate shapes
    if len(x_train.shape) != 3 or len(x_val.shape) != 3 or len(x_test.shape) != 3:
        raise ValueError(
            f"Expected 3D arrays, got shapes: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    # Reshape and fit on training data only
    x = np.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))

    # Calculate mean and std from training data
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    # Normalize training data
    x = (x - mean) / std
    x_train = np.reshape(x, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Free memory
    del x
    gc.collect()

    # Transform validation data using training statistics
    x = np.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
    x = (x - mean) / std
    x_val = np.reshape(x, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))

    del x
    gc.collect()

    # Transform test data using training statistics
    x = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2]))
    x = (x - mean) / std
    x_test = np.reshape(x, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    del x
    gc.collect()

    return x_train, x_val, x_test


def norma_pre(x_all):
    """Normalize data using z-score normalization"""
    if len(x_all.shape) != 3:
        raise ValueError(f"Expected 3D array (samples, timesteps, features), got shape {x_all.shape}")

    x = np.reshape(x_all, (x_all.shape[0] * x_all.shape[1], x_all.shape[2]))

    # Calculate mean and std for each feature
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    # Normalize
    x = (x - mean) / std

    x_all = np.reshape(x, (x_all.shape[0], x_all.shape[1], x_all.shape[2]))

    del x
    gc.collect()

    return x_all


def aug_data(x_train, y_train, transformations, sigma_l, ext, batch_size=100):
    """
    Apply data augmentation transformations with batching to reduce memory

    Args:
        batch_size: Process this many samples at a time
    """
    window_size = x_train.shape[1]
    num_sample = x_train.shape[0]
    if ext:
        m_ = len(transformations) + 1
    else:
        m_ = 2

    # Pre-allocate arrays
    x_train_pro = np.zeros((len(transformations), num_sample * m_, window_size, x_train.shape[-1]), dtype=np.float32)
    y_train_pro = np.zeros((len(transformations), num_sample * m_), dtype=bool)

    # Process in batches to reduce memory pressure
    for batch_start in range(0, num_sample, batch_size):
        batch_end = min(batch_start + batch_size, num_sample)

        for j in range(batch_start, batch_end):
            x_train_temp = np.copy(x_train[j])
            for Jt, sigma, i in zip(transformations, sigma_l, range(len(transformations))):
                x_train_pro[i, j * m_, :, :] = x_train_temp
                y_train_pro[i, j * m_] = False
                x_train_pro[i, j * m_ + 1, :, :] = Jt(x_train_temp, sigma=sigma)
                y_train_pro[i, j * m_ + 1] = True
                if ext:
                    cnt = 1
                    for k in range(len(transformations)):
                        if i != k:
                            x_train_pro[i, j * m_ + 1 + cnt, :, :] = transformations[k](x_train_temp, sigma=sigma_l[k])
                            y_train_pro[i, j * m_ + 1 + cnt] = False
                            cnt += 1

        if (batch_end) % 1000 == 0:
            print(f"Augmented {batch_end}/{num_sample} samples...")
            gc.collect()

    print(x_train_pro.shape)
    print(y_train_pro.shape)
    return x_train_pro, y_train_pro