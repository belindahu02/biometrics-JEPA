import pandas as pd
import numpy as np
import os
import mne
import gc


def data_load_eeg(path, users, frame_size=30):
    """
    Load EEG data from EDF files with session-level splitting to prevent data leakage.
    Sessions 1-10: training, Sessions 11-12: validation, Sessions 13-14: testing

    Memory-optimized version: processes data in smaller chunks
    """
    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []
    sessions = []

    for user_id, user in enumerate(users):
        user_folder = f"S{user:03d}"
        user_path = os.path.join(path, user_folder)
        count = 0

        if not os.path.exists(user_path):
            print(f"Warning: User folder {user_folder} not found")
            sessions.append(0)
            continue

        # Process each session (R01 to R14)
        for session in range(1, 15):
            filename = f"S{user:03d}R{session:02d}.edf"
            filepath = os.path.join(user_path, filename)

            try:
                # Load EDF file
                raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

                # Get data from all EEG channels
                data = raw.get_data().T  # Shape: (n_samples, n_channels)

                # Clean up raw object immediately
                del raw
                gc.collect()

                # Create overlapping windows
                if data.shape[0] < frame_size:
                    continue

                # Truncate to multiple of frame_size for consistent windowing
                data = data[:(data.shape[0] // frame_size) * frame_size]

                # Create sliding windows with 50% overlap
                windowed_data = np.lib.stride_tricks.sliding_window_view(
                    data, (frame_size, data.shape[1])
                )[::frame_size // 2, :]
                windowed_data = windowed_data.reshape(
                    windowed_data.shape[0], windowed_data.shape[2], windowed_data.shape[3]
                )

                # Clean up intermediate data
                del data

                # Convert to float32 to save memory (default is float64)
                windowed_data = windowed_data.astype(np.float32)

                # Session-level split to prevent data leakage
                if session <= 10:  # Training sessions
                    x_train.append(windowed_data)
                    y_train.extend([user_id] * windowed_data.shape[0])
                elif session <= 12:  # Validation sessions
                    x_val.append(windowed_data)
                    y_val.extend([user_id] * windowed_data.shape[0])
                else:  # Testing sessions (13-14)
                    x_test.append(windowed_data)
                    y_test.extend([user_id] * windowed_data.shape[0])

                count += 1

            except (FileNotFoundError, ValueError, RuntimeError) as e:
                continue

        sessions.append(count)

        # Periodic garbage collection
        if (user_id + 1) % 10 == 0:
            gc.collect()

    # Concatenate all data
    if x_train:
        x_train = np.concatenate(x_train, axis=0).astype(np.float32)
    else:
        x_train = np.array([]).astype(np.float32)

    if x_val:
        x_val = np.concatenate(x_val, axis=0).astype(np.float32)
    else:
        x_val = np.array([]).astype(np.float32)

    if x_test:
        x_test = np.concatenate(x_test, axis=0).astype(np.float32)
    else:
        x_test = np.array([]).astype(np.float32)

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Testing samples: {len(y_test)}")

    return x_train, np.array(y_train), x_val, np.array(y_val), x_test, np.array(y_test), sessions


def norma(x_train, x_val, x_test):
    """
    Normalize data using mean and standard deviation from training data only.
    Memory-optimized version using float32.
    """
    # Ensure float32 type
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Reshape for normalization
    original_train_shape = x_train.shape
    original_val_shape = x_val.shape
    original_test_shape = x_test.shape

    x_train_flat = np.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))

    # Calculate mean and std from training data only
    mean = np.mean(x_train_flat, axis=0, dtype=np.float32)
    std = np.std(x_train_flat, axis=0, dtype=np.float32)

    # Avoid division by zero
    std[std == 0] = 1.0

    # Normalize training data
    x_train_flat = (x_train_flat - mean) / std
    x_train = np.reshape(x_train_flat, original_train_shape)
    del x_train_flat
    gc.collect()

    # Transform validation data
    if x_val.size > 0:
        x_val_flat = np.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
        x_val_flat = (x_val_flat - mean) / std
        x_val = np.reshape(x_val_flat, original_val_shape)
        del x_val_flat
        gc.collect()

    # Transform test data
    if x_test.size > 0:
        x_test_flat = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2]))
        x_test_flat = (x_test_flat - mean) / std
        x_test = np.reshape(x_test_flat, original_test_shape)
        del x_test_flat
        gc.collect()

    return x_train, x_val, x_test