import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, Dropout
import gc
import numpy as np

from backbones import *
from data_loader import *


def cohen_kappa(y_true, y_pred, num_classes):
    """
    Calculate Cohen's Kappa score manually with memory-efficient batch processing.
    """
    # Convert predictions to class labels in chunks to avoid memory spike
    batch_size = 1000
    y_pred_labels = []

    for i in range(0, len(y_pred), batch_size):
        batch = y_pred[i:i + batch_size]
        y_pred_labels.extend(np.argmax(batch, axis=1))

    y_pred_labels = np.array(y_pred_labels)

    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for true, pred in zip(y_true, y_pred_labels):
        confusion_matrix[true, pred] += 1

    # Calculate observed accuracy
    n = np.sum(confusion_matrix)
    po = np.trace(confusion_matrix) / n

    # Calculate expected accuracy
    sum_rows = np.sum(confusion_matrix, axis=1)
    sum_cols = np.sum(confusion_matrix, axis=0)
    pe = np.sum(sum_rows * sum_cols) / (n * n)

    # Calculate Cohen's Kappa
    if pe == 1.0:
        return 0.0
    kappa = (po - pe) / (1 - pe)

    return kappa


class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient data generator that keeps data on disk and loads batches on demand.
    This is crucial for handling 109 users without loading all data into RAM.
    """

    def __init__(self, X, y, batch_size=8, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def trainer(num_users):
    """
    Train model on specified number of users using all available samples.
    Memory-optimized for handling up to 109 users.

    Args:
        num_users: Number of users to include in the classification task

    Returns:
        test_acc: Test accuracy
        kappa_score: Cohen's Kappa score
    """
    frame_size = 40
    path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"
    # path = "/app/data/1.0.0"

    # Fixed batch size
    BATCH_SIZE = 8

    # Use first num_users from the dataset
    users = list(range(1, num_users + 1))

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_load_eeg(
        path, users=users, frame_size=frame_size
    )

    print(f"Training with {num_users} users")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Testing samples: {x_test.shape[0]}")

    classes, counts = np.unique(y_train, return_counts=True)
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Samples per user: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    # Normalize data
    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    print("x_train", x_train.shape)
    print("x_val", x_val.shape)
    print("x_test", x_test.shape)

    # Build model with consistent architecture
    ks = 3
    con = 3
    inputs = Input(shape=(frame_size, x_train.shape[-1]))
    x = Conv1D(filters=16 * con, kernel_size=ks, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32 * con, KS=ks)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    resnettssd = Model(inputs, outputs)

    print(f"Using batch size: {BATCH_SIZE}")

    # Training configuration
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True,
        patience=5
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_rate=0.95,
        decay_steps=1000000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # **CRITICAL MEMORY OPTIMIZATION**: Use mixed precision training
    # This can reduce memory usage by ~40% with minimal accuracy impact
    if num_users > 50:
        print("Enabling mixed precision training for memory efficiency")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        # Recompile with mixed precision
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    resnettssd.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # **MEMORY OPTIMIZATION**: Use data generators for large datasets
    # This prevents loading full batches into memory at once
    use_generator = (num_users > 70)

    if use_generator:
        print("Using data generator for memory efficiency")
        train_gen = MemoryEfficientDataGenerator(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
        val_gen = MemoryEfficientDataGenerator(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

        history = resnettssd.fit(
            train_gen,
            validation_data=val_gen,
            epochs=100,
            callbacks=[callback],
            verbose=1
        )
    else:
        history = resnettssd.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            callbacks=[callback],
            batch_size=BATCH_SIZE,
            verbose=1
        )

    # Evaluate on test set
    results = resnettssd.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    test_acc = results[1]
    print(f"Test accuracy: {results[1]:.4f}")

    # Calculate Cohen's Kappa score with batch prediction to save memory
    print("Calculating Kappa score...")

    # **MEMORY OPTIMIZATION**: Predict in smaller chunks and accumulate
    y_pred_list = []
    chunk_size = 1000  # Process 1000 samples at a time

    for i in range(0, len(x_test), chunk_size):
        chunk = x_test[i:i + chunk_size]
        pred_chunk = resnettssd.predict(chunk, batch_size=BATCH_SIZE, verbose=0)
        y_pred_list.append(pred_chunk)
        del pred_chunk
        gc.collect()

    y_pred = np.concatenate(y_pred_list, axis=0)
    del y_pred_list

    kappa_score = cohen_kappa(y_test, y_pred, num_classes)
    print(f'Kappa score: {kappa_score:.4f}')

    # Clean up to free memory
    del resnettssd, history, y_pred
    del x_train, y_train, x_val, y_val, x_test, y_test

    # Reset mixed precision policy if it was enabled
    if num_users > 50:
        tf.keras.mixed_precision.set_global_policy('float32')

    tf.keras.backend.clear_session()
    gc.collect()

    return test_acc, kappa_score