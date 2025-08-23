# =============================================
# data_loader_no_sklearn.py
# =============================================

import os
import numpy as np


def pca_reduce(X, n_components):
    """
    Manual PCA using NumPy (SVD-based).

    Args:
        X: array of shape (samples, features)
        n_components: number of principal components to keep

    Returns:
        Reduced data of shape (samples, n_components), explained variance ratio
    """
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]
    components = eigvecs[:, :n_components]
    explained_variance_ratio = eigvals[:n_components] / np.sum(eigvals)
    X_reduced = np.dot(X_centered, components)
    return X_reduced, explained_variance_ratio


def spectrogram_to_1d_conversion(spectrograms, method='pca', target_features=24):
    batch_size, time_steps, freq_bins = spectrograms.shape
    print(f"Converting spectrograms from {spectrograms.shape} using method: {method}")

    if method == 'pca':
        reshaped = spectrograms.reshape(-1, freq_bins)
        reduced, evr = pca_reduce(reshaped, target_features)
        result = reduced.reshape(batch_size, time_steps, target_features)
        print(f"PCA explained variance ratio: {np.sum(evr):.3f}")

    elif method == 'downsample':
        indices = np.linspace(0, freq_bins - 1, target_features, dtype=int)
        result = spectrograms[:, :, indices]
        print(f"Downsampled frequency bins: {indices}")

    elif method == 'average_bands':
        band_size = freq_bins // target_features
        bands = []
        for i in range(target_features):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, freq_bins)
            band_avg = np.mean(spectrograms[:, :, start_idx:end_idx], axis=2)
            bands.append(band_avg)
        result = np.stack(bands, axis=2)

    elif method == 'mel_bands':
        mel_indices = np.logspace(0, np.log10(freq_bins - 1), target_features, dtype=int)
        mel_indices = np.unique(mel_indices)
        if len(mel_indices) < target_features:
            linear_indices = np.linspace(0, freq_bins - 1, target_features, dtype=int)
            result = spectrograms[:, :, linear_indices]
        else:
            result = spectrograms[:, :, mel_indices[:target_features]]
        print(f"Mel-scale indices: {mel_indices[:target_features]}")

    else:
        raise ValueError(f"Unknown conversion method: {method}")

    print(f"Converted to shape: {result.shape}")
    return result


def load_spectrogram_data_as_1d(data_path, user_ids, conversion_method='pca', target_features=24):
    """
    Load spectrogram data and convert to 1D format (pure NumPy, no sklearn).

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, sessions
    """
    print(f"Loading spectrogram data for {len(user_ids)} users...")
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
        for item in sorted(os.listdir(user_path)):
            session_path = os.path.join(user_path, item)
            if os.path.isdir(session_path) and item.startswith(user_folder + 'R'):
                spec_file = os.path.join(session_path, "001_stacked.npy")
                if os.path.exists(spec_file):
                    try:
                        spec = np.load(spec_file)
                        # if spec.shape == (4160, 768):
                        if spec.shape == (520, 768):
                            user_spectrograms.append(spec)
                            session_count += 1
                        else:
                            print(f"Warning: {spec_file} has shape {spec.shape}, expected (520, 768)")
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

    X = np.vstack(all_spectrograms)
    y = np.concatenate(all_labels)
    print(f"Total spectrograms loaded: {X.shape[0]}")
    print(f"Original spectrogram shape: {X.shape[1:]}")
    X_1d = spectrogram_to_1d_conversion(X, method=conversion_method, target_features=target_features)

    # Split per user (70% train, 15% val, 15% test)
    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []

    for user_idx in np.unique(y):
        user_mask = y == user_idx
        user_data = X_1d[user_mask]
        user_labels = y[user_mask]

        n_samples = len(user_data)
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.15)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        x_train_list.append(user_data[train_idx])
        y_train_list.append(user_labels[train_idx])
        x_val_list.append(user_data[val_idx])
        y_val_list.append(user_labels[val_idx])
        x_test_list.append(user_data[test_idx])
        y_test_list.append(user_labels[test_idx])

    x_train = np.vstack(x_train_list)
    y_train = np.concatenate(y_train_list)
    x_val = np.vstack(x_val_list)
    y_val = np.concatenate(y_val_list)
    x_test = np.vstack(x_test_list)
    y_test = np.concatenate(y_test_list)

    print(f"Split completed: train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")
    return x_train, y_train, x_val, y_val, x_test, y_test, sessions


def user_data_split(x, y, samples_per_user):
    """Limit number of samples per user"""
    users = np.unique(y)
    x_train, y_train = np.array([]), np.array([])
    for user in users:
        idx = np.where(y == user)[0]
        np.random.shuffle(idx)
        idx = idx[:samples_per_user]
        if x_train.shape[0] == 0:
            x_train = x[idx]
            y_train = y[idx]
        else:
            x_train = np.concatenate((x_train, x[idx]), axis=0)
            y_train = np.concatenate((y_train, y[idx]), axis=0)
    return x_train, y_train

def norma(x_train, x_val, x_test):
    """
    Normalize 1D converted data (zero mean, unit variance) without sklearn
    """
    flat_train = x_train.reshape(-1, x_train.shape[-1])
    mean = np.mean(flat_train, axis=0)
    std = np.std(flat_train, axis=0) + 1e-8

    def normalize(x):
        flat = x.reshape(-1, x.shape[-1])
        return ((flat - mean) / std).reshape(x.shape)

    return normalize(x_train), normalize(x_val), normalize(x_test)
