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


class AugmentedDataGenerator(tf.keras.utils.Sequence):
    """
    Wrapper generator that applies data augmentation on top of EEGDataGenerator.
    """

    def __init__(self, base_generator, augmentation_fn=None):
        """
        Args:
            base_generator: EEGDataGenerator instance
            augmentation_fn: Function to apply augmentation (e.g., tf_magwarp)
        """
        self.base_generator = base_generator
        self.augmentation_fn = augmentation_fn

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, batch_idx):
        X_batch, y_batch = self.base_generator[batch_idx]

        if self.augmentation_fn is not None:
            # Apply augmentation to each sample in batch
            X_augmented = []
            for i in range(X_batch.shape[0]):
                aug_sample = self.augmentation_fn(X_batch[i]).numpy()
                X_augmented.append(aug_sample)
            X_batch = np.array(X_augmented, dtype=np.float32)

        return X_batch, y_batch

    def on_epoch_end(self):
        self.base_generator.on_epoch_end()


def evaluate_with_generator(model, generator, num_classes):
    """
    Evaluate model using a generator and compute accuracy and kappa.

    Args:
        model: Trained model
        generator: Data generator
        num_classes: Number of classes

    Returns:
        accuracy, kappa_score
    """
    y_true_list = []
    y_pred_list = []

    print("Evaluating model...")
    for batch_idx in range(len(generator)):
        X_batch, y_batch = generator[batch_idx]
        y_pred_batch = model.predict(X_batch, verbose=0)

        y_true_list.append(y_batch)
        y_pred_list.append(y_pred_batch)

        if (batch_idx + 1) % 100 == 0:
            print(f"Evaluated {batch_idx + 1}/{len(generator)} batches")

    # Concatenate all predictions
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Calculate accuracy
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_true == y_pred_classes)

    # Calculate kappa
    kappa = cohen_kappa_score(y_true, y_pred, num_classes)

    return accuracy, kappa


def trainer(num_users):
    """
    Train model on specified number of users from EEG MMI dataset.
    Uses streaming/generator approach to avoid memory issues.

    Args:
        num_users: Number of users to include in classification task

    Returns:
        test_acc: Test accuracy
        kappa_score: Cohen's Kappa score
    """
    frame_size = 40
    BATCH_SIZE = 8

    path = "/app/data/1.0.0"
    # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"

    # Use first num_users users
    users = list(range(1, num_users + 1))

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users")
    print(f"{'=' * 60}")

    # Load data using streaming approach
    train_files, val_files, test_files, mean, std, n_channels, train_samples, val_samples, test_samples = \
        data_load_eeg_mmi_streaming(path, users=users, frame_size=frame_size)

    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Testing samples: {test_samples}")

    num_classes = num_users
    print(f"Number of classes: {num_classes}")
    print(f"Number of channels: {n_channels}")

    # Create data generators
    print("Creating data generators...")
    train_generator = EEGDataGenerator(
        train_files,
        frame_size,
        mean,
        std,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Wrap training generator with augmentation
    train_generator_aug = AugmentedDataGenerator(
        train_generator,
        augmentation_fn=tf_magwarp
    )

    val_generator = EEGDataGenerator(
        val_files,
        frame_size,
        mean,
        std,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_generator = EEGDataGenerator(
        test_files,
        frame_size,
        mean,
        std,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Train batches: {len(train_generator_aug)}")
    print(f"Val batches: {len(val_generator)}")
    print(f"Test batches: {len(test_generator)}")

    # Build model
    ks = 3
    con = 3
    inputs = Input(shape=(frame_size, n_channels))
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

    print("\nStarting training...")
    history = resnettssd.fit(
        train_generator_aug,
        validation_data=val_generator,
        epochs=100,
        callbacks=[callback],
        verbose=1
    )

    # Evaluate on test set using generator
    print("\nEvaluating on test set...")
    test_acc, kappa_score = evaluate_with_generator(resnettssd, test_generator, num_classes)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Kappa score: {kappa_score:.4f}")

    # Aggressive cleanup
    del resnettssd, history, optimizer, lr_schedule
    del train_generator, train_generator_aug, val_generator, test_generator
    del train_files, val_files, test_files

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