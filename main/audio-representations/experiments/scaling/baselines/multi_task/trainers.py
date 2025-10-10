from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

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


def trainer(num_users, fet_extrct, scen, ft):
    """
    Train classifier with specified number of users.

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

    frame_size = 30
    # path = "/app/data/1.0.0"
    path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data"  # Update this path

    # Select users based on num_users parameter
    users = list(range(1, num_users + 1))

    print(f"\n{'=' * 60}")
    print(f"Training with {num_users} users")
    print(f"{'=' * 60}")

    # Load data with session-level splitting
    x_train, y_train, x_val, y_val, x_test, y_test, sessions_train = data_load(
        path, users=users, frame_size=30
    )

    print("Training samples:", x_train.shape[0])
    print("Validation samples:", x_val.shape[0])
    print("Testing samples:", x_test.shape[0])

    classes, counts = np.unique(y_train, return_counts=True)
    num_classes = len(classes)
    print("Number of classes:", num_classes)
    print("Samples per class:", counts)

    # Normalize data
    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    print("x_train", x_train.shape)
    print("x_val", x_val.shape)
    print("x_test", x_test.shape)

    # Build classifier on top of feature extractor
    inputs = Input(shape=(frame_size, x_train.shape[-1]))
    x = fet_extrct(inputs, training=False)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    resnettssd = Model(inputs, outputs)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', restore_best_weights=True, patience=5
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001 / (ft + 1), decay_rate=0.95, decay_steps=1000
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
        callbacks=callback,
        batch_size=8,
        verbose=1
    )

    results = resnettssd.evaluate(x_test, y_test)
    test_acc = results[1]
    print("Test accuracy:", results[1])

    # Calculate kappa score
    metric = compute_cohen_kappa(num_classes=num_classes, sparse_labels=True)
    metric.update_state(y_true=y_test, y_pred=resnettssd.predict(x_test))
    result = metric.result()
    kappa_score = result.numpy()
    print('Kappa score:', result.numpy())

    return test_acc, kappa_score