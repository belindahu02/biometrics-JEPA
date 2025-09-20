import pandas as pd
import numpy as np
import os
import mne
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

def apply_masking(data, frame_size, num_masks=2):

    """
    Apply random masking to a frame of data.

    Args:
        data: Frame data of shape (frame_size, n_channels)
        frame_size: Size of the frame
        num_masks: Number of masks to apply per frame

    Returns:
        masked_data: Data with masks applied
    """

    masked_data = data.copy()

    for _ in range(num_masks):
        # Random start position within the frame
        start_pos = np.random.randint(0, frame_size)

        # Random mask size between 1.25% and 5% of frame size
        min_mask_size = max(1, int(frame_size * 0.0125))
        max_mask_size = max(min_mask_size, int(frame_size * 0.05))
        mask_size = np.random.randint(min_mask_size, max_mask_size + 1)

        # Ensure mask doesn't exceed frame boundaries
        end_pos = min(start_pos + mask_size, frame_size)

        # Apply mask by setting values to zero (you can modify this masking strategy)
        masked_data[start_pos:end_pos, :] = 0

    return masked_data


def load_edf_file(filepath):
    """
    Load EDF file using MNE and return the data.

    Args:
        filepath: Path to EDF file

    Returns:
        data: EEG data array of shape (n_samples, n_channels)
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        # Get data and transpose to (n_samples, n_channels)
        data = raw.get_data().T
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def data_load_origin(path, users, folders, frame_size=30):
    """
    Load EEG MMI dataset with masking applied to each frame.

    Args:
        path: Root path to dataset
        users: List of user IDs (e.g., [1, 2, 3, ...])
        folders: Not used in EEG MMI dataset (kept for compatibility)
        frame_size: Size of each frame in samples

    Returns:
        x_train: Training data with applied masks
        y_train: User labels
        sessions: Number of sessions per user
    """
    sessions = []
    x_train = np.array([])
    y_train = []

    for user_id, user in enumerate(users):
        count = 0
        user_folder = f"S{user:03d}"  # Format as S001, S002, etc.
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"User folder {user_path} not found")
            sessions.append(0)
            continue

        # Find all EDF files for this user
        edf_files = [f for f in os.listdir(user_path) if f.endswith('.edf')]

        for edf_file in edf_files:
            filepath = os.path.join(user_path, edf_file)

            # Load EDF file
            data = load_edf_file(filepath)

            if data is None:
                continue

            # Create sliding windows
            if data.shape[0] < frame_size:
                continue

            # Create overlapping windows with stride of frame_size//2
            windows = np.lib.stride_tricks.sliding_window_view(
                data, (frame_size, data.shape[1])
            )[::frame_size // 2, :]

            # Reshape to (n_windows, frame_size, n_channels)
            windows = windows.reshape(windows.shape[0], windows.shape[2], windows.shape[3])

            # Apply masking to each frame
            masked_windows = np.array([
                apply_masking(window, frame_size) for window in windows
            ])

            if x_train.shape[0] == 0:
                x_train = masked_windows
                y_train += [user_id] * masked_windows.shape[0]
            else:
                x_train = np.concatenate((x_train, masked_windows), axis=0)
                y_train += [user_id] * masked_windows.shape[0]

            count += 1

        sessions.append(count)
        print(f"User {user}: loaded {count} files")

    print(f"Total data shape: {x_train.shape}")
    return x_train, np.array(y_train), sessions


def norma_origin(x_all):
    """
    Normalize data using z-score normalization (mean=0, std=1).
    """
    # Reshape to (samples*timesteps, features)
    x = np.reshape(x_all, (x_all.shape[0] * x_all.shape[1], x_all.shape[2]))

    # Calculate mean and std for each feature
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    # Normalize
    x = (x - mean) / std

    # Reshape back to original shape
    x_all = np.reshape(x, (x_all.shape[0], x_all.shape[1], x_all.shape[2]))
    return x_all


def user_data_split(x, y, samples_per_user):
    """
    Split data ensuring equal samples per user.
    """
    users, counts = np.unique(y, return_counts=True)
    x_train = np.array([])
    y_train = np.array([])

    for user in users:
        indx = np.where(y == user)[0]
        np.random.shuffle(indx)
        indx = indx[:min(samples_per_user, len(indx))]

        if x_train.shape[0] == 0:
            x_train = x[indx]
            y_train = y[indx]
        else:
            x_train = np.concatenate((x_train, x[indx]), axis=0)
            y_train = np.concatenate((y_train, y[indx]), axis=0)

    return x_train, y_train


def data_load(path, users, frame_size=30):
    """
    Load EEG MMI dataset with train/val/test split and masking.

    Args:
        path: Root path to dataset
        users: List of user IDs
        frame_size: Size of each frame in samples

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, sessions
    """
    sessions = []
    x_train = np.array([])
    x_val = np.array([])
    x_test = np.array([])
    y_train = []
    y_val = []
    y_test = []

    for user_id, user in enumerate(users):
        count = 0
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"User folder {user_path} not found")
            sessions.append(0)
            continue

        # Find all EDF files for this user
        edf_files = [f for f in os.listdir(user_path) if f.endswith('.edf')]

        user_data = []

        for edf_file in edf_files:
            filepath = os.path.join(user_path, edf_file)
            data = load_edf_file(filepath)

            if data is None:
                continue

            if data.shape[0] < frame_size:
                continue

            user_data.append(data)
            count += 1

        if not user_data:
            sessions.append(0)
            continue

        # Concatenate all data for this user
        all_user_data = np.concatenate(user_data, axis=0)

        # Split into train/val/test (70%/15%/15%)
        total_samples = all_user_data.shape[0]
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)

        train_data = all_user_data[:train_end]
        val_data = all_user_data[train_end:val_end]
        test_data = all_user_data[val_end:]

        # Create windows for each split
        for data_split, x_split, y_split in [
            (train_data, 'train', y_train),
            (val_data, 'val', y_val),
            (test_data, 'test', y_test)
        ]:
            if data_split.shape[0] < frame_size:
                continue

            windows = np.lib.stride_tricks.sliding_window_view(
                data_split, (frame_size, data_split.shape[1])
            )[::frame_size // 2, :]

            windows = windows.reshape(windows.shape[0], windows.shape[2], windows.shape[3])

            # Apply masking
            masked_windows = np.array([
                apply_masking(window, frame_size) for window in windows
            ])

            # Add to appropriate split
            if x_split == 'train':
                if x_train.shape[0] == 0:
                    x_train = masked_windows
                    y_train += [user_id] * masked_windows.shape[0]
                else:
                    x_train = np.concatenate((x_train, masked_windows), axis=0)
                    y_train += [user_id] * masked_windows.shape[0]
            elif x_split == 'val':
                if x_val.shape[0] == 0:
                    x_val = masked_windows
                    y_val += [user_id] * masked_windows.shape[0]
                else:
                    x_val = np.concatenate((x_val, masked_windows), axis=0)
                    y_val += [user_id] * masked_windows.shape[0]
            else:  # test
                if x_test.shape[0] == 0:
                    x_test = masked_windows
                    y_test += [user_id] * masked_windows.shape[0]
                else:
                    x_test = np.concatenate((x_test, masked_windows), axis=0)
                    y_test += [user_id] * masked_windows.shape[0]

        sessions.append(count)
        print(f"User {user}: loaded {count} files")

    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")
    return x_train, np.array(y_train), x_val, np.array(y_val), x_test, np.array(y_test), sessions


def norma(x_train, x_val, x_test):
    """
    Normalize train/val/test splits using z-score normalization fit on training data.
    """
    # Fit normalization parameters on training data
    x = np.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    # Normalize training data
    x = (x - mean) / std
    x_train = np.reshape(x, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Normalize validation data using training stats
    x = np.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
    x = (x - mean) / std
    x_val = np.reshape(x, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))

    # Normalize test data using training stats
    x = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2]))
    x = (x - mean) / std
    x_test = np.reshape(x, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    return x_train, x_val, x_test


# Example usage:
"""
# Load data for users S001 to S010
users = list(range(1, 11))
path = "/path/to/eeg_mmi_dataset"
frame_size = 160  # Adjust based on your sampling rate

# Load with train/val/test split
x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_load(path, users, frame_size)

# Normalize data
x_train, x_val, x_test = norma(x_train, x_val, x_test)

print(f"Training data: {x_train.shape}")
print(f"Validation data: {x_val.shape}")  
print(f"Test data: {x_test.shape}")
"""
