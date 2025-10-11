import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
import gc

from backbones import *
from data_loader import *
from transformations_tf import *


def cohen_kappa_score(y_true, y_pred, num_classes):
    """
    Calculate Cohen's Kappa score without sklearn or tensorflow_addons.

    Args:
        y_true: True labels (1D array)
        y_pred: Predicted labels (1D array or 2D probabilities)
        num_classes: Number of classes

    Returns:
        kappa: Cohen's Kappa score
    """
    # Convert predictions to class labels if needed
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Build confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1

    # Calculate observed accuracy
    n = len(y_true)
    observed_accuracy = np.trace(confusion_matrix) / n

    # Calculate expected accuracy
    expected_accuracy = 0
    for i in range(num_classes):
        expected_accuracy += (np.sum(confusion_matrix[i, :]) * np.sum(confusion_matrix[:, i])) / (n * n)

    # Calculate Cohen's Kappa
    if expected_accuracy == 1.0:
        kappa = 1.0
    else:
        kappa = (observed_accuracy - expected_accuracy) / (1.0 - expected_accuracy)

    return kappa


def trainer(num_users):
    """
    Train model on specified number of users from EEG MMI dataset.
    Memory-efficient version with aggressive cleanup.

    Args:
        num_users: Number of users to include in classification task

    Returns:
        test_acc: Test accuracy
        kappa_score: Cohen's Kappa score
    """
    frame_size = 40
    BATCH_SIZE = 8
    AUTO = tf.data.AUTOTUNE

    path = "/app/data/1.0.0"
    # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"

    # Use first num_users users
    users = list(range(1, num_users + 1))

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users")
    print(f"{'=' * 60}")

    # Load data with session-based splitting
    x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_load_eeg_mmi(
        path, users=users, frame_size=frame_size
    )

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Testing samples: {x_test.shape[0]}")

    classes, counts = np.unique(y_train, return_counts=True)
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Minimum samples per user: {min(counts)}")

    # Normalize data
    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    print(f"x_train: {x_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"x_test: {x_test.shape}")

    print(f"Using 100% of training samples: {x_train.shape[0]}")

    # Create datasets with optimized memory settings
    SEED = 34

    # Training dataset - don't cache to save memory
    ds_x = tf.data.Dataset.from_tensor_slices(x_train)
    ds_x = (
        ds_x.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
            .map(tf_magwarp, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(2)  # Reduced prefetch buffer
    )

    ds_y = tf.data.Dataset.from_tensor_slices(y_train)
    ds_y = (
        ds_y.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(2)
    )
    ssl_ds = tf.data.Dataset.zip((ds_x, ds_y))

    # Validation dataset - smaller prefetch
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

    # Build model
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

    # Compile and train
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True,
        patience=5
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_rate=0.95,
        decay_steps=1000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    resnettssd.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = resnettssd.fit(
        ssl_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[callback],
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Evaluate on test set
    results = resnettssd.evaluate(x_test, y_test, verbose=0, batch_size=BATCH_SIZE)
    test_acc = results[1]
    print(f"Test accuracy: {test_acc:.4f}")

    # Calculate Cohen's Kappa score - predict in smaller batches to save memory
    y_pred = resnettssd.predict(x_test, verbose=0, batch_size=BATCH_SIZE)
    kappa_score = cohen_kappa_score(y_test, y_pred, num_classes)
    print(f"Kappa score: {kappa_score:.4f}")

    # Aggressive cleanup
    del resnettssd, history, optimizer, lr_schedule
    del ssl_ds, ds_x, ds_y, val_ds
    del x_train, y_train, x_val, y_val, x_test, y_test
    del y_pred

    # Clear TensorFlow session
    tf.keras.backend.clear_session()

    # Force garbage collection
    gc.collect()

    # Clear GPU memory if available
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass

    return test_acc, kappa_score