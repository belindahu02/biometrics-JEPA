import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, Dropout, MaxPooling1D
import gc
import numpy as np

from backbones import *
from data_loader import StreamingEEGDataGenerator, calculate_normalization_stats


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


def evaluate_with_streaming_generator(model, generator, num_classes):
    """
    Evaluate model using streaming generator and calculate metrics.
    """
    print(f"Evaluating on {len(generator.sample_index)} samples...")

    # Evaluate accuracy
    results = model.evaluate(generator, verbose=0)
    test_acc = results[1]
    print(f"Test accuracy: {test_acc:.4f}")

    # Calculate Kappa with streaming predictions
    print("Calculating Kappa score...")
    y_true_all = []
    y_pred_all = []

    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        y_pred_batch = model.predict(X_batch, batch_size=len(X_batch), verbose=0)

        y_true_all.extend(y_batch)
        y_pred_all.append(y_pred_batch)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(generator)} batches")
            gc.collect()

    y_true_all = np.array(y_true_all)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    kappa_score = cohen_kappa(y_true_all, y_pred_all, num_classes)
    print(f'Kappa score: {kappa_score:.4f}')

    del y_true_all, y_pred_all
    gc.collect()

    return test_acc, kappa_score


def trainer(num_users):
    """
    Train model on specified number of users using streaming data loader.
    This version never loads all data into memory at once.

    Args:
        num_users: Number of users to include in the classification task

    Returns:
        test_acc: Test accuracy
        kappa_score: Cohen's Kappa score
    """
    frame_size = 40
    # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"
    path = "/app/data/1.0.0"

    BATCH_SIZE = 8

    # Use first num_users from the dataset
    users = list(range(1, num_users + 1))
    num_classes = num_users

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users (streaming mode)")
    print(f"{'=' * 60}\n")

    # Step 1: Calculate normalization statistics from training data
    normalization_stats = calculate_normalization_stats(
        path, users, frame_size=frame_size, max_samples=10000
    )

    # Step 2: Create streaming data generators
    print("\nCreating data generators...")

    train_generator = StreamingEEGDataGenerator(
        path=path,
        users=users,
        split='train',
        frame_size=frame_size,
        batch_size=BATCH_SIZE,
        shuffle=True,
        normalization_stats=normalization_stats
    )

    val_generator = StreamingEEGDataGenerator(
        path=path,
        users=users,
        split='val',
        frame_size=frame_size,
        batch_size=BATCH_SIZE,
        shuffle=False,
        normalization_stats=normalization_stats
    )

    test_generator = StreamingEEGDataGenerator(
        path=path,
        users=users,
        split='test',
        frame_size=frame_size,
        batch_size=BATCH_SIZE,
        shuffle=False,
        normalization_stats=normalization_stats
    )

    print(f"\nDataset summary:")
    print(f"  Training samples: {len(train_generator.sample_index)}")
    print(f"  Validation samples: {len(val_generator.sample_index)}")
    print(f"  Testing samples: {len(test_generator.sample_index)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Get number of channels from first batch
    sample_batch, _ = train_generator[0]
    n_channels = sample_batch.shape[-1]
    print(f"  Number of channels: {n_channels}")

    # Step 3: Build model
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
    model = Model(inputs, outputs)

    # Step 4: Configure training
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True,
        patience=5,
        verbose=1
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_rate=0.95,
        decay_steps=1000000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Enable mixed precision for large-scale experiments
    if num_users > 50:
        print("Enabling mixed precision training")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 5: Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=[callback],
        verbose=1,
    )

    # Step 6: Evaluate
    print("\nEvaluating model...")
    test_acc, kappa_score = evaluate_with_streaming_generator(
        model, test_generator, num_classes
    )

    # Cleanup
    del model, history, train_generator, val_generator, test_generator
    del normalization_stats

    if num_users > 50:
        tf.keras.mixed_precision.set_global_policy('float32')

    tf.keras.backend.clear_session()
    gc.collect()

    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Kappa Score: {kappa_score:.4f}")

    return test_acc, kappa_score