import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, Dropout

from backbones import *
from data_loader import *


def cohen_kappa(y_true, y_pred, num_classes):
    """
    Calculate Cohen's Kappa score manually.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (probabilities)
        num_classes: Number of classes

    Returns:
        kappa: Cohen's Kappa score
    """
    # Convert predictions to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)

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


def trainer(num_users):
    """
    Train model on specified number of users using all available samples.

    Args:
        num_users: Number of users to include in the classification task

    Returns:
        test_acc: Test accuracy
        kappa_score: Cohen's Kappa score
    """
    frame_size = 30
    path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"  # Update this path

    # Use first num_users from the dataset
    users = list(range(1, num_users + 1))

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
    print(f"Samples per user: {counts}")

    # Normalize data
    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    print("x_train", x_train.shape)
    print("x_val", x_val.shape)
    print("x_test", x_test.shape)

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

    resnettssd.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = resnettssd.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=[callback],
        batch_size=8
    )

    # Evaluate on test set
    results = resnettssd.evaluate(x_test, y_test)
    test_acc = results[1]
    print(f"Test accuracy: {results[1]}")

    # Calculate Cohen's Kappa score
    y_pred = resnettssd.predict(x_test)
    kappa_score = cohen_kappa(y_test, y_pred, num_classes)
    print(f'Kappa score: {kappa_score}')

    return test_acc, kappa_score