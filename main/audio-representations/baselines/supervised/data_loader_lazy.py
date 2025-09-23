import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
import mne
import warnings
from typing import List, Tuple, Generator
import gc

# Suppress MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


def apply_masking(data, frame_size, num_masks=2):
    """Apply random masking to a frame of data."""
    masked_data = data.copy()
    for _ in range(num_masks):
        start_pos = np.random.randint(0, frame_size)
        min_mask_size = max(1, int(frame_size * 0.0125))
        max_mask_size = max(min_mask_size, int(frame_size * 0.05))
        mask_size = np.random.randint(min_mask_size, max_mask_size + 1)
        end_pos = min(start_pos + mask_size, frame_size)
        masked_data[start_pos:end_pos, :] = 0
    return masked_data


def load_edf_file(filepath):
    """Load EDF file using MNE and return the data."""
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data = raw.get_data().T
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


class EEGDataGenerator(Sequence):
    """Memory-efficient data generator that loads data on-demand."""

    def __init__(self, file_paths, user_labels, frame_size=30, batch_size=32,
                 samples_per_user=None, shuffle=True, apply_masking_flag=True):
        """
        Initialize the data generator.

        Args:
            file_paths: List of tuples (user_id, filepath)
            user_labels: List of user labels corresponding to file_paths
            frame_size: Size of each frame
            batch_size: Batch size for training
            samples_per_user: Limit samples per user (None for no limit)
            shuffle: Whether to shuffle data
            apply_masking_flag: Whether to apply masking
        """
        self.file_paths = file_paths
        self.user_labels = user_labels
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.samples_per_user = samples_per_user
        self.shuffle = shuffle
        self.apply_masking_flag = apply_masking_flag

        # Pre-compute normalization statistics
        self.mean = None
        self.std = None

        # Build index of available windows
        self._build_window_index()

        if self.shuffle:
            self._shuffle_indices()

    def _build_window_index(self):
        """Build an index of all available windows without loading data."""
        self.window_index = []  # List of (user_id, file_idx, window_start, window_end)

        user_window_counts = {}

        for user_id, filepath in self.file_paths:
            # Load file to get dimensions
            data = load_edf_file(filepath)
            if data is None or data.shape[0] < self.frame_size:
                continue

            # Calculate number of windows
            num_windows = (data.shape[0] - self.frame_size) // (self.frame_size // 2) + 1

            # Add windows to index
            for i in range(num_windows):
                start_idx = i * (self.frame_size // 2)
                end_idx = start_idx + self.frame_size

                if end_idx <= data.shape[0]:
                    # Limit samples per user if specified
                    if self.samples_per_user is None or user_window_counts.get(user_id, 0) < self.samples_per_user:
                        self.window_index.append((user_id, filepath, start_idx, end_idx))
                        user_window_counts[user_id] = user_window_counts.get(user_id, 0) + 1

            # Clear data from memory
            del data
            gc.collect()

        print(f"Total windows indexed: {len(self.window_index)}")

    def _shuffle_indices(self):
        """Shuffle the window indices."""
        np.random.shuffle(self.window_index)

    def compute_normalization_stats(self, sample_size=1000):
        """Compute normalization statistics from a sample of the data."""
        print("Computing normalization statistics...")

        sample_indices = np.random.choice(len(self.window_index),
                                          min(sample_size, len(self.window_index)),
                                          replace=False)

        all_samples = []

        for idx in sample_indices:
            user_id, filepath, start_idx, end_idx = self.window_index[idx]

            # Load only the required window
            data = load_edf_file(filepath)
            if data is not None:
                window = data[start_idx:end_idx]
                all_samples.append(window.reshape(-1, window.shape[-1]))

            del data
            gc.collect()

        if all_samples:
            combined_samples = np.concatenate(all_samples, axis=0)
            self.mean = np.mean(combined_samples, axis=0)
            self.std = np.std(combined_samples, axis=0)
            self.std[self.std == 0] = 1.0

            del all_samples, combined_samples
            gc.collect()

            print("Normalization statistics computed.")
        else:
            print("Warning: No samples found for normalization.")
            self.mean = 0
            self.std = 1

    def __len__(self):
        """Number of batches per epoch."""
        return len(self.window_index) // self.batch_size

    def __getitem__(self, idx):
        """Get a batch of data."""
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.window_index))

        batch_x = []
        batch_y = []

        for i in range(batch_start, batch_end):
            user_id, filepath, start_idx, end_idx = self.window_index[i]

            # Load only the required window
            data = load_edf_file(filepath)
            if data is not None:
                window = data[start_idx:end_idx]

                # Apply masking if enabled
                if self.apply_masking_flag:
                    window = apply_masking(window, self.frame_size)

                # Normalize
                if self.mean is not None and self.std is not None:
                    window = (window - self.mean) / self.std

                batch_x.append(window)
                batch_y.append(user_id)

            del data
            gc.collect()

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            self._shuffle_indices()


def create_file_index(path, users):
    """Create an index of all EDF files for the given users."""
    file_paths = []
    user_labels = []

    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            print(f"User folder {user_path} not found")
            continue

        edf_files = [f for f in os.listdir(user_path) if f.endswith('.edf')]

        for edf_file in edf_files:
            filepath = os.path.join(user_path, edf_file)
            file_paths.append((user_id, filepath))
            user_labels.append(user_id)

    return file_paths, user_labels


def split_files_by_user(file_paths, train_ratio=0.7, val_ratio=0.15):
    """Split files by user for train/val/test."""
    user_files = {}

    # Group files by user
    for user_id, filepath in file_paths:
        if user_id not in user_files:
            user_files[user_id] = []
        user_files[user_id].append(filepath)

    train_files = []
    val_files = []
    test_files = []

    for user_id, files in user_files.items():
        np.random.shuffle(files)
        n_files = len(files)

        train_end = int(n_files * train_ratio)
        val_end = int(n_files * (train_ratio + val_ratio))

        train_user_files = [(user_id, f) for f in files[:train_end]]
        val_user_files = [(user_id, f) for f in files[train_end:val_end]]
        test_user_files = [(user_id, f) for f in files[val_end:]]

        train_files.extend(train_user_files)
        val_files.extend(val_user_files)
        test_files.extend(test_user_files)

    return train_files, val_files, test_files


# Memory-efficient trainer function matching original logic
def memory_efficient_trainer(samples_per_user=None, batch_size=8, frame_size=30):
    """Memory-efficient training function that matches original logic."""

    # Set path and users - SAME AS ORIGINAL
    path = "/app/1.0.0/"
    users_2 = list(range(1, 110))  # Same variable name as original

    # Create separate generators for train/val/test folders - MATCHES ORIGINAL SPLIT
    train_files, _ = create_file_index(os.path.join(path, "TrainingSet"), users_2)
    val_files, _ = create_file_index(os.path.join(path, "TestingSet"), users_2)
    test_files, _ = create_file_index(os.path.join(path, "TestingSet_secret"), users_2)

    print(f"training samples files: {len(train_files)}")
    print(f"validation samples files: {len(val_files)}")
    print(f"testing samples files: {len(test_files)}")

    # Create generators
    train_generator = EEGDataGenerator(
        train_files, [f[0] for f in train_files],
        frame_size=frame_size, batch_size=batch_size,
        samples_per_user=None, shuffle=True  # Don't limit here initially
    )

    val_generator = EEGDataGenerator(
        val_files, [f[0] for f in val_files],
        frame_size=frame_size, batch_size=batch_size,
        samples_per_user=None, shuffle=False
    )

    test_generator = EEGDataGenerator(
        test_files, [f[0] for f in test_files],
        frame_size=frame_size, batch_size=batch_size,
        samples_per_user=None, shuffle=False
    )

    # Compute normalization statistics from training data - MATCHES ORIGINAL
    train_generator.compute_normalization_stats()
    val_generator.mean = train_generator.mean
    val_generator.std = train_generator.std
    test_generator.mean = train_generator.mean
    test_generator.std = train_generator.std

    # Check minimum samples per user - MATCHES ORIGINAL LOGIC
    user_counts = {}
    for user_id, _ in train_files:
        user_counts[user_id] = user_counts.get(user_id, 0) + 1

    classes = list(user_counts.keys())
    counts = list(user_counts.values())
    num_classes = len(classes)
    print("minimum samples per user:", min(counts) if counts else 0)

    # Apply samples_per_user limit if specified - MATCHES ORIGINAL
    if samples_per_user is not None:
        train_generator.samples_per_user = samples_per_user
        train_generator._build_window_index()  # Rebuild with limit
        print(f"limited training samples: {len(train_generator.window_index)}")

    # Build EXACT same model as original
    from backbones import resnetblock_final  # Import your function

    ks = 3
    con = 3
    inputs = Input(shape=(frame_size, train_generator.window_index[0][1].split('/')[-1].endswith(
        '.edf') and 64 or 64))  # You'll need to get actual channel count
    x = Conv1D(filters=16 * con, kernel_size=ks, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32 * con, KS=ks)  # NOW INCLUDED
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    resnettssd = Model(inputs, outputs)  # Same variable name

    # EXACT same compilation as original
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_rate=0.95, decay_steps=1000000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train with SAME parameters
    history = resnettssd.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=[callback],
        # batch_size is handled by generator
    )

    # Evaluate EXACTLY like original
    results = resnettssd.evaluate(test_generator)
    test_acc = results[1]
    print("test acc:", results[1])

    # Calculate Cohen's Kappa - SAME AS ORIGINAL
    y_pred_probs = resnettssd.predict(test_generator)

    # Get actual test labels for kappa calculation
    y_test = []
    for i in range(len(test_generator)):
        _, batch_y = test_generator[i]
        y_test.extend(batch_y)
    y_test = np.array(y_test)

    kappa_score = compute_cohen_kappa(y_test, y_pred_probs, num_classes)
    print('kappa score:', kappa_score)

    return test_acc, kappa_score


# Alternative: Simple memory-efficient data loader
def load_data_streaming(path, users, frame_size=30, max_samples_per_user=100):
    """
    Stream data loading - yields batches of data instead of loading all at once.
    """
    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)

        if not os.path.exists(user_path):
            continue

        edf_files = [f for f in os.listdir(user_path) if f.endswith('.edf')]
        sample_count = 0

        for edf_file in edf_files:
            if sample_count >= max_samples_per_user:
                break

            filepath = os.path.join(user_path, edf_file)
            data = load_edf_file(filepath)

            if data is None or data.shape[0] < frame_size:
                continue

            # Create windows
            num_windows = (data.shape[0] - frame_size) // (frame_size // 2) + 1

            for i in range(num_windows):
                if sample_count >= max_samples_per_user:
                    break

                start_idx = i * (frame_size // 2)
                end_idx = start_idx + frame_size

                if end_idx <= data.shape[0]:
                    window = data[start_idx:end_idx]
                    window = apply_masking(window, frame_size)
                    yield window, user_id
                    sample_count += 1

            del data
            gc.collect()


# Memory-efficient version of your original trainer
def memory_efficient_trainer_v2(samples_per_user=100):
    """
    Memory-efficient version that processes data in smaller chunks.
    """
    frame_size = 30
    path = "/app/1.0.0/"
    users = list(range(1, 110))
    num_classes = len(users)

    # Collect normalization statistics first
    print("Computing normalization statistics...")
    sample_data = []
    sample_count = 0
    max_norm_samples = 1000

    for window, _ in load_data_streaming(path, users, frame_size, max_samples_per_user=10):
        sample_data.append(window.reshape(-1, window.shape[-1]))
        sample_count += 1
        if sample_count >= max_norm_samples:
            break

    if sample_data:
        combined_samples = np.concatenate(sample_data, axis=0)
        mean = np.mean(combined_samples, axis=0)
        std = np.std(combined_samples, axis=0)
        std[std == 0] = 1.0
        del sample_data, combined_samples
        gc.collect()
    else:
        mean, std = 0, 1

    # Build and compile model
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout

    # Get input shape from first sample
    for window, _ in load_data_streaming(path, users, frame_size, max_samples_per_user=1):
        input_shape = window.shape
        break

    ks = 3
    con = 3
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=16 * con, kernel_size=ks, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training with mini-batches
    batch_size = 32
    epochs = 100

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Collect a batch of data
        batch_x = []
        batch_y = []

        for window, label in load_data_streaming(path, users, frame_size, samples_per_user):
            # Normalize
            normalized_window = (window - mean) / std
            batch_x.append(normalized_window)
            batch_y.append(label)

            # Train when batch is full
            if len(batch_x) >= batch_size:
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)

                model.train_on_batch(batch_x, batch_y)

                # Clear batch
                batch_x = []
                batch_y = []

                # Force garbage collection
                gc.collect()

        # Train on remaining samples
        if batch_x:
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            model.train_on_batch(batch_x, batch_y)

    return model