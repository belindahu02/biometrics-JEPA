import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
import gc

from backbones import *
from data_loader import *


def compute_cohen_kappa(y_true, y_pred, num_classes):
    """
    Compute Cohen's Kappa score manually.

    Args:
        y_true: True labels (1D array)
        y_pred: Predicted class probabilities (2D array) or class indices (1D array)
        num_classes: Number of classes

    Returns:
        Cohen's Kappa score
    """
    # Convert predictions to class indices if needed
    if len(y_pred.shape) == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred_classes):
        confusion_matrix[int(true), int(pred)] += 1

    # Calculate observed agreement (accuracy)
    n = np.sum(confusion_matrix)
    observed_agreement = np.trace(confusion_matrix) / n

    # Calculate expected agreement
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    expected_agreement = np.sum(row_sums * col_sums) / (n * n)

    # Calculate Cohen's Kappa
    if expected_agreement == 1.0:
        return 1.0  # Perfect agreement

    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return kappa


def trainer(num_users, fet_extrct, scen, ft, config=None):
    """
    Train classifier with specified number of users using data generators.

    Args:
        num_users: Number of users to include in classification task
        fet_extrct: Pre-trained feature extractor
        scen: Scenario number
        ft: Fine-tuning configuration (0-5)
        config: Memory configuration dict with frame_size and max_samples_per_session
    """
    if config is None:
        config = {
            'frame_size': 40,  # Must match pre-trained feature extractor
            'max_samples_per_session': None
        }

    ft_dict = {0: 17, 1: 12, 2: 11, 3: 8, 4: 5, 5: 0}
    ft = ft_dict[ft]

    # Set feature extractor trainability
    for i in range(1, ft + 1):
        fet_extrct.layers[i].trainable = False

    frame_size = config['frame_size']
    path = "/app/data/1.0.0"  # Update this path as needed

    # Select users based on num_users parameter
    users = list(range(1, num_users + 1))
    num_classes = num_users  # Each user is a class

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users")
    print(f"{'=' * 60}")

    # Fixed batch size of 8
    batch_size = 8

    # Load data with generators - this uses session-level splitting
    train_gen, val_gen, test_gen, steps = data_load_with_generators(
        path,
        users=users,
        frame_size=frame_size,
        batch_size=batch_size,
        max_samples_per_session=config.get('max_samples_per_session')
    )

    print(f"Training steps per epoch: {steps['train']}")
    print(f"Validation steps per epoch: {steps['val']}")
    print(f"Test steps per epoch: {steps['test']}")
    print(f"Number of classes: {num_classes}")

    # Get input shape from first batch
    sample_batch = next(iter(train_gen))
    input_shape = sample_batch[0].shape[1:]  # (frame_size, n_channels)
    print(f"Input shape: {input_shape}")

    # Convert generators to tf.data.Dataset
    def create_dataset(generator, steps_per_epoch):
        """Convert generator to tf.data.Dataset"""
        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=(None, frame_size, input_shape[-1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        return dataset

    train_dataset = create_dataset(train_gen, steps['train'])
    val_dataset = create_dataset(val_gen, steps['val'])
    test_dataset = create_dataset(test_gen, steps['test'])

    # Build classifier on top of feature extractor

    inputs = Input(shape=input_shape)
    x = fet_extrct(inputs, training=False)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    resnettssd = Model(inputs, outputs)

    # Callbacks with memory management
    class MemoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                gc.collect()

    callback_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', restore_best_weights=True, patience=5
    )
    callback_memory = MemoryCallback()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001 / (ft + 1), decay_rate=0.95, decay_steps=1000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    resnettssd.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with datasets
    history = resnettssd.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=steps['train'],
        validation_steps=steps['val'],
        epochs=100,
        callbacks=[callback_early, callback_memory],
        verbose=1
    )

    # Evaluate with dataset
    results = resnettssd.evaluate(test_dataset, steps=steps['test'], verbose=1)
    test_acc = results[1]
    print("Test accuracy:", test_acc)

    # Calculate kappa score - need to collect predictions from dataset
    print("Computing Cohen's Kappa score...")
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in test_dataset:
        y_pred_batch = resnettssd.predict(batch_x, verbose=0)
        y_true_all.extend(batch_y.numpy())
        y_pred_all.append(y_pred_batch)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    kappa_score = compute_cohen_kappa(y_true_all, y_pred_all, num_classes)
    print('Kappa score:', kappa_score)

    # Clean up
    del y_true_all, y_pred_all
    gc.collect()

    return test_acc, kappa_score