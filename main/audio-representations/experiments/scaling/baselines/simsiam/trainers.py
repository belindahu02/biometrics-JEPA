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


class CachedGeneratorSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence that caches batches from a generator for efficient training
    """

    def __init__(self, generator, batches_to_cache=None):
        self.generator = generator
        self.batches_to_cache = batches_to_cache
        self.cached_batches = []
        self.cache_data()

    def cache_data(self):
        """Cache batches from the generator"""
        self.cached_batches = []
        batch_count = 0

        for batch_x, batch_y in self.generator:
            self.cached_batches.append((batch_x, batch_y))
            batch_count += 1

            if self.batches_to_cache is not None and batch_count >= self.batches_to_cache:
                break

    def __len__(self):
        return len(self.cached_batches)

    def __getitem__(self, idx):
        return self.cached_batches[idx]

    def on_epoch_end(self):
        """Re-cache data at the end of each epoch"""
        self.cache_data()
        gc.collect()


def trainer(num_users, fet_extrct, scen, ft):
    """
    Train classifier with specified number of users using generators.

    Args:
        num_users: Number of users to include in classification task
        fet_extrct: Pre-trained feature extractor
        scen: Scenario number
        ft: Fine-tuning configuration (0-5)
    """
    ft_dict = {0: 17, 1: 12, 2: 11, 3: 8, 4: 5, 5: 0}
    ft = ft_dict[ft]

    # Set feature extractor trainability
    for i in range(1, ft + 1):
        fet_extrct.layers[i].trainable = False

    frame_size = 40
    # path = "/app/data/1.0.0"
    path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"  # Update this path

    batch_size = 8  # Small batch size to manage memory

    # Select users based on num_users parameter
    # users = list(range(1, num_users + 1))
    users = list(range(1, 2))

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users")
    print(f"{'=' * 60}")

    # Create generators with proper normalization
    train_gen, val_gen, test_gen, steps = data_load_with_generators(
        path, users=users, frame_size=frame_size,
        batch_size=batch_size,
        max_samples_per_session=10000  # Limit to manage memory
    )

    print(f"Steps per epoch - Train: {steps['train']}, Val: {steps['val']}, Test: {steps['test']}")

    # Get data shape and number of classes from first batch
    first_batch_x, first_batch_y = next(iter(train_gen))
    n_channels = first_batch_x.shape[-1]
    num_classes = len(np.unique(first_batch_y))

    print(f"Data shape: (batch_size, {frame_size}, {n_channels})")
    print(f"Number of classes: {num_classes}")

    # Build classifier on top of feature extractor
    inputs = Input(shape=(frame_size, n_channels))
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

    # Create Keras Sequences from generators
    print("\nCaching training data...")
    train_sequence = CachedGeneratorSequence(train_gen, batches_to_cache=steps['train'])

    print("Caching validation data...")
    val_sequence = CachedGeneratorSequence(val_gen, batches_to_cache=steps['val'])

    print("Caching test data...")
    test_sequence = CachedGeneratorSequence(test_gen, batches_to_cache=steps['test'])

    print("\nStarting training...")
    history = resnettssd.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=100,
        callbacks=[callback_early, callback_memory],
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = resnettssd.evaluate(test_sequence, verbose=1)
    test_acc = test_results[1]

    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Calculate kappa score - collect all predictions
    print("Computing predictions for Kappa score...")
    all_y_pred = resnettssd.predict(test_sequence, verbose=1)

    # Get true labels from test sequence
    all_y_true = []
    for i in range(len(test_sequence)):
        _, y_batch = test_sequence[i]
        all_y_true.extend(y_batch)
    all_y_true = np.array(all_y_true)

    kappa_score = compute_cohen_kappa(all_y_true, all_y_pred, num_classes)
    print(f'Kappa score: {kappa_score:.4f}')

    # Clean up
    del all_y_true, all_y_pred, train_sequence, val_sequence, test_sequence
    gc.collect()

    return test_acc, kappa_score