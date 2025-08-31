# =============================================
# data_loader_2d.py
# =============================================

import os
import numpy as np


def normalize_spectrogram(spectrograms, method='log_scale'):
    """
    Normalize 2D spectrograms using various methods with proper handling of edge cases.

    Args:
        spectrograms: array of shape (batch, time, freq)
        method: normalization method ('log_scale', 'min_max', 'z_score', 'decibel')

    Returns:
        Normalized spectrograms
    """
    if method == 'log_scale':
        # Ensure all values are positive before log transform
        # Add small epsilon and take absolute value to handle negatives
        spectrograms_positive = np.abs(spectrograms) + 1e-8
        result = np.log(spectrograms_positive)

        # Check for any remaining invalid values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(
                f"Warning: Found invalid values after log transform. Min: {np.min(spectrograms)}, Max: {np.max(spectrograms)}")
            # Replace invalid values with small negative number
            result = np.where(np.isfinite(result), result, -10.0)

        return result

    elif method == 'decibel':
        # Convert to decibel scale with proper handling of zeros/negatives
        spectrograms_positive = np.abs(spectrograms) + 1e-8
        result = 10 * np.log10(spectrograms_positive)

        # Check for invalid values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Warning: Found invalid values after decibel transform.")
            result = np.where(np.isfinite(result), result, -80.0)  # -80 dB as floor

        return result

    elif method == 'min_max':
        # Min-max normalization per sample
        batch_size = spectrograms.shape[0]
        normalized = np.zeros_like(spectrograms)
        for i in range(batch_size):
            spec = spectrograms[i]
            min_val = np.min(spec)
            max_val = np.max(spec)

            # Handle case where min == max (constant spectrogram)
            if max_val - min_val < 1e-8:
                normalized[i] = np.zeros_like(spec)
            else:
                normalized[i] = (spec - min_val) / (max_val - min_val)
        return normalized

    elif method == 'z_score':
        # Z-score normalization per sample
        batch_size = spectrograms.shape[0]
        normalized = np.zeros_like(spectrograms)
        for i in range(batch_size):
            spec = spectrograms[i]
            mean = np.mean(spec)
            std = np.std(spec)

            # Handle case where std is zero (constant spectrogram)
            if std < 1e-8:
                normalized[i] = np.zeros_like(spec)
            else:
                normalized[i] = (spec - mean) / std
        return normalized

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_spectrogram_data_2d(data_path, user_ids, normalization='log_scale',
                             add_channel_dim=True):
    """
    Load 2D spectrogram data without compression.

    Args:
        data_path: Path to the dataset
        user_ids: List of user IDs to load
        normalization: Method for normalizing spectrograms
        add_channel_dim: Whether to add channel dimension for CNN input

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, sessions, data_info
    """
    print(f"Loading 2D spectrogram data for {len(user_ids)} users...")
    all_spectrograms = []
    all_labels = []
    sessions = []

    for user_idx, user_id in enumerate(user_ids):
        user_folder = f"S{user_id:03d}"
        user_path = os.path.join(data_path, user_folder)
        if not os.path.exists(user_path):
            print(f"Warning: Path {user_path} does not exist")
            sessions.append(0)
            continue

        user_spectrograms = []
        session_count = 0

        # Loop over session folders like S001R01, S001R02, etc.
        for item in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, item)
            if os.path.isdir(session_path) and item.startswith(user_folder + 'R'):
                # Loop over all stacked files in the session folder
                for npy_file in sorted(os.listdir(session_path)):
                    if npy_file.endswith("_stacked.npy"):
                        spec_file = os.path.join(session_path, npy_file)
                        try:
                            spec = np.load(spec_file)
                            # Handle both possible shapes
                            if spec.shape in [(4160, 768), (520, 768)]:
                                user_spectrograms.append(spec)
                                session_count += 1
                            else:
                                print(f"Warning: {spec_file} has shape {spec.shape}")
                        except Exception as e:
                            print(f"Error loading {spec_file}: {e}")

        if len(user_spectrograms) == 0:
            print(f"No valid spectrograms for user {user_id}")
            sessions.append(0)
            continue

        user_spectrograms = np.array(user_spectrograms)
        user_labels = np.full(len(user_spectrograms), user_idx)
        all_spectrograms.append(user_spectrograms)
        all_labels.append(user_labels)
        sessions.append(session_count)
        print(f"User {user_id}: {len(user_spectrograms)} spectrograms loaded")

    if len(all_spectrograms) == 0:
        raise ValueError("No spectrograms loaded for any user")

    X = np.vstack(all_spectrograms)
    y = np.concatenate(all_labels)

    print(f"Total spectrograms loaded: {X.shape[0]}")
    print(f"Original spectrogram shape: {X.shape[1:]} (time, frequency)")

    print(f"Spectrogram value range: min={np.min(X)}, max={np.max(X)}")
    print(f"Number of negative values: {np.sum(X < 0)}")
    print(f"Number of zero values: {np.sum(X == 0)}")

    # Apply normalization
    if normalization:
        print(f"Applying {normalization} normalization...")
        X = normalize_spectrogram(X, method=normalization)

    # Add channel dimension for CNN input: (batch, height, width) -> (batch, 1, height, width)
    if add_channel_dim:
        X = X[:, np.newaxis, :, :]  # Add channel dimension
        print(f"Added channel dimension: {X.shape}")

    # Split per user (70% train, 15% val, 15% test)
    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []

    for user_idx in np.unique(y):
        user_mask = y == user_idx
        user_data = X[user_mask]
        user_labels = y[user_mask]

        n_samples = len(user_data)
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.15)
        n_test = n_samples - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n_samples - n_val - n_test

        # Random permutation for splitting
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        if len(train_idx) > 0:
            x_train_list.append(user_data[train_idx])
            y_train_list.append(user_labels[train_idx])
        if len(val_idx) > 0:
            x_val_list.append(user_data[val_idx])
            y_val_list.append(user_labels[val_idx])
        if len(test_idx) > 0:
            x_test_list.append(user_data[test_idx])
            y_test_list.append(user_labels[test_idx])

    # Combine all splits
    x_train = np.vstack(x_train_list) if x_train_list else np.empty((0,) + X.shape[1:])
    y_train = np.concatenate(y_train_list) if y_train_list else np.empty(0, dtype=int)
    x_val = np.vstack(x_val_list) if x_val_list else np.empty((0,) + X.shape[1:])
    y_val = np.concatenate(y_val_list) if y_val_list else np.empty(0, dtype=int)
    x_test = np.vstack(x_test_list) if x_test_list else np.empty((0,) + X.shape[1:])
    y_test = np.concatenate(y_test_list) if y_test_list else np.empty(0, dtype=int)

    # Data information
    data_info = {
        'original_shape': X.shape[1:],
        'normalization': normalization,
        'num_users': len(user_ids),
        'total_samples': X.shape[0],
        'sessions_per_user': sessions
    }

    print(f"Split completed:")
    print(f"  Train: {x_train.shape}")
    print(f"  Val: {x_val.shape}")
    print(f"  Test: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test, sessions, data_info


def user_data_split_2d(x, y, samples_per_user):
    """
    Limit number of samples per user for 2D data.
    """
    users = np.unique(y)
    x_limited = []
    y_limited = []

    for user in users:
        idx = np.where(y == user)[0]
        np.random.shuffle(idx)
        idx = idx[:samples_per_user]

        x_limited.append(x[idx])
        y_limited.append(y[idx])

    return np.vstack(x_limited), np.concatenate(y_limited)


def global_normalize_2d(x_train, x_val, x_test):
    """
    Apply global normalization across the entire training set.
    Useful for standardizing the input range across all samples.
    """
    if x_train.ndim == 4:  # (batch, channel, height, width)
        # Calculate statistics across all dimensions except batch
        mean = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(x_train, axis=(0, 2, 3), keepdims=True) + 1e-8
    else:  # (batch, height, width)
        mean = np.mean(x_train)
        std = np.std(x_train) + 1e-8

    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std
    x_test_norm = (x_test - mean) / std

    return x_train_norm, x_val_norm, x_test_norm


def augment_spectrograms(spectrograms, augmentations=['time_mask', 'freq_mask']):
    """
    Apply data augmentation to spectrograms.

    Args:
        spectrograms: array of shape (batch, [channel], time, freq)
        augmentations: list of augmentation types

    Returns:
        Augmented spectrograms
    """
    augmented = spectrograms.copy()

    for aug in augmentations:
        if aug == 'time_mask':
            # Random time masking
            for i in range(len(augmented)):
                if np.random.rand() > 0.5:  # 50% chance
                    spec = augmented[i]
                    time_dim = -2 if spec.ndim == 3 else -2  # Works for both 3D and 4D
                    t_max = spec.shape[time_dim]
                    mask_size = np.random.randint(1, min(t_max // 10, 50))
                    start = np.random.randint(0, t_max - mask_size)
                    if spec.ndim == 3:  # (channel, time, freq)
                        spec[:, start:start + mask_size, :] = 0
                    else:  # (time, freq)
                        spec[start:start + mask_size, :] = 0

        elif aug == 'freq_mask':
            # Random frequency masking
            for i in range(len(augmented)):
                if np.random.rand() > 0.5:  # 50% chance
                    spec = augmented[i]
                    freq_dim = -1
                    f_max = spec.shape[freq_dim]
                    mask_size = np.random.randint(1, min(f_max // 10, 50))
                    start = np.random.randint(0, f_max - mask_size)
                    if spec.ndim == 3:  # (channel, time, freq)
                        spec[:, :, start:start + mask_size] = 0
                    else:  # (time, freq)
                        spec[:, start:start + mask_size] = 0

        elif aug == 'noise':
            # Add random noise
            noise_std = 0.01
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = augmented + noise

    return augmented