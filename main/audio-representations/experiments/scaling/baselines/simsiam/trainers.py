import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
import gc
import os

from backbones import *
from data_loader import *


def cohen_kappa_score(y_true, y_pred, num_classes):
    """
  Manual implementation of Cohen's Kappa score
  """
    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[int(true), int(pred)] += 1

    n = np.sum(confusion_matrix)

    # Observed agreement
    po = np.trace(confusion_matrix) / n

    # Expected agreement
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    pe = np.sum(row_sum * col_sum) / (n * n)

    # Cohen's Kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)

    return kappa


def trainer(num_users, fet_extrct, scen, ft, checkpoint_dir="checkpoints"):
    ft_dict = {0: 17, 1: 12, 2: 11, 3: 8, 4: 5, 5: 0}
    ft = ft_dict[ft]

    # Freeze layers
    for i in range(1, ft + 1):
        fet_extrct.layers[i].trainable = False

    frame_size = 40
    # path = "/app/data/1.0.0"
    path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"  # Update this path

    # Use first num_users for the classification task
    users = list(range(1, num_users + 1))

    # Load data with session-level splitting
    folder_train = ["TrainingSet"]
    folder_val = ["TestingSet"]
    folder_test = ["TestingSet_secret"]

    print(f"\nLoading data for {num_users} users...")

    # Memory efficient: limit samples during loading if needed
    max_samples_per_user = None  # Set to a number (e.g., 1000) if still hitting memory issues

    x_train, y_train, sessions_train = data_load_origin(path, users=users, folders=folder_train,
                                                        frame_size=frame_size,
                                                        max_samples_per_user=max_samples_per_user)
    print(f"Training samples: {x_train.shape[0]}")

    x_val, y_val, sessions_val = data_load_origin(path, users=users, folders=folder_val,
                                                  frame_size=frame_size, max_samples_per_user=max_samples_per_user)
    print(f"Validation samples: {x_val.shape[0]}")

    x_test, y_test, sessions_test = data_load_origin(path, users=users, folders=folder_test,
                                                     frame_size=frame_size, max_samples_per_user=max_samples_per_user)
    print(f"Testing samples: {x_test.shape[0]}")

    if x_train.shape[0] == 0 or x_val.shape[0] == 0 or x_test.shape[0] == 0:
        print(f"Warning: Insufficient data for {num_users} users")
        return 0.0, 0.0

    classes, counts = np.unique(y_train, return_counts=True)
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Samples per class - min: {min(counts)}, max: {max(counts)}, mean: {np.mean(counts):.1f}")

    # Normalize data
    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    print(f"x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")

    # Build classification model
    inputs = Input(shape=(frame_size, x_train.shape[-1]))
    x = fet_extrct(inputs, training=False)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    resnettssd = Model(inputs, outputs)

    # Callbacks
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(checkpoint_dir, f"model_users{num_users}_temp.weights.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True,
        patience=5
    )

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_rate=0.95, decay_steps=1000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = resnettssd.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=[early_stopping, checkpoint_callback],
        batch_size=8,
        verbose=1
    )

    # Evaluate
    results = resnettssd.evaluate(x_test, y_test, verbose=0)
    test_acc = results[1]
    print(f"Test accuracy: {test_acc:.4f}")

    # Calculate kappa score in batches to save memory
    batch_size = 32
    y_pred_all = []

    for i in range(0, len(x_test), batch_size):
        batch = x_test[i:i + batch_size]
        y_pred_batch = resnettssd.predict(batch, verbose=0)
        y_pred_all.append(np.argmax(y_pred_batch, axis=1))
        del y_pred_batch
        gc.collect()

    y_pred_classes = np.concatenate(y_pred_all)
    kappa_score = cohen_kappa_score(y_test, y_pred_classes, num_classes)
    print(f'Kappa score: {kappa_score:.4f}')

    # Clean up
    del resnettssd, x_train, y_train, x_val, y_val, x_test, y_test
    del y_pred_classes, y_pred_all
    tf.keras.backend.clear_session()
    gc.collect()

    # Remove temporary checkpoint
    if os.path.exists(model_checkpoint_path):
        os.remove(model_checkpoint_path)

    return test_acc, kappa_score