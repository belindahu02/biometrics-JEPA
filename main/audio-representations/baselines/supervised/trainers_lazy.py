import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
from tensorflow.keras import layers
import os
import json
import pickle
from datetime import datetime

from backbones import *
from data_loader_lazy import *

# Base directory setup
base_dir = "/disks/SATA_2/belinda_h/data"
graph_data_dir = os.path.join(base_dir, "supervised/graph_data")
graphs_dir = os.path.join(base_dir, "supervised/graphs")

# Make sure these exist
os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)


def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Manual implementation of train_test_split without sklearn.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def compute_roc_curve(y_true, y_scores):
    """
    Manual ROC curve computation without sklearn.
    """
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    thresholds = y_scores[threshold_idxs]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    fpr = fps / fps[-1] if fps[-1] > 0 else np.repeat(np.nan, fps.shape)
    tpr = tps / tps[-1] if tps[-1] > 0 else np.repeat(np.nan, tps.shape)

    return fpr, tpr, thresholds


def compute_tsne(X, n_components=2, perplexity=30, n_iter=1000, learning_rate=200.0, random_state=None):
    """
    Basic t-SNE implementation without sklearn.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    X_squared = np.sum(X ** 2, axis=1, keepdims=True)
    distances = X_squared + X_squared.T - 2 * np.dot(X, X.T)
    distances = np.maximum(distances, 0)

    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0
        for _ in range(50):
            H = np.exp(-distances[i] * beta)
            H[i] = 0
            sum_H = np.sum(H)
            if sum_H == 0:
                H = np.ones_like(H)
                sum_H = n_samples - 1
            P_i = H / sum_H
            entropy = -np.sum(P_i * np.log2(P_i + 1e-12))
            perp = 2 ** entropy
            if abs(perp - perplexity) < 1e-5:
                break
            if perp > perplexity:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
        P[i] = P_i

    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)

    Y = np.random.normal(0, 1e-4, (n_samples, n_components))
    for iter in range(n_iter):
        sum_Y = np.sum(Y ** 2, axis=1, keepdims=True)
        num = 1 / (1 + sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        PQ = P - Q
        for i in range(n_samples):
            dY_i = np.sum((PQ[i, :, np.newaxis] * num[i, :, np.newaxis] * (Y[i] - Y)), axis=0)
            Y[i] = Y[i] - learning_rate * dY_i
        Y = Y - np.mean(Y, axis=0)
        if iter == 100:
            learning_rate = learning_rate / 4
    return Y


def compute_cohen_kappa(y_true, y_pred, num_classes):
    """
    Manual Cohen's Kappa implementation (no tensorflow_addons).
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.argmax(y_pred, axis=1).astype(int)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    n_samples = np.sum(confusion)
    po = np.trace(confusion) / n_samples
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (n_samples ** 2)

    kappa = (po - pe) / (1 - pe + 1e-12)
    return kappa


def trainer(samples_per_user):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(graph_data_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== TRAINING RUN STARTED: {timestamp} ===")
    print(f"Run directory: {run_dir}")
    print(f"Samples per user: {samples_per_user}")

    frame_size = 30
    path = "/app/1.0.0/"
    batch_size = 5  # Process users in smaller batches for memory efficiency

    users_2 = list(range(1, 110))

    print(f"\n=== LOADING DATA ===")
    print("Using NEW data loader with test-only masking")

    # Use the new data_load function instead of data_load_origin
    x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_load(
        path=path, users=users_2, frame_size=frame_size, batch_size=batch_size
    )

    print("training samples : ", x_train.shape[0])
    print("validation samples : ", x_val.shape[0])
    print("testing samples : ", x_test.shape[0])

    # Save data shapes info
    data_info = {
        'timestamp': timestamp,
        'frame_size': frame_size,
        'batch_size': batch_size,
        'total_users': len(users_2),
        'samples_per_user': samples_per_user,
        'original_shapes': {
            'train': list(x_train.shape),
            'val': list(x_val.shape),
            'test': list(x_test.shape)
        },
        'sessions': sessions
    }

    classes, counts = np.unique(y_train, return_counts=True)
    num_classes = len(classes)
    print("minimum samples per user : ", min(counts))

    print(f"\n=== NORMALIZATION ===")
    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    print("x_train", x_train.shape)
    print("x_val", x_val.shape)
    print("x_test", x_test.shape)

    print(f"\n=== USER DATA SPLITTING ===")
    x_train, y_train = user_data_split(x_train, y_train, samples_per_user=samples_per_user)
    print("limited training samples : ", x_train.shape[0])
    classes, counts = np.unique(y_train, return_counts=True)
    print(counts)

    # Update data_info with final shapes
    data_info['final_shapes'] = {
        'train': list(x_train.shape),
        'val': list(x_val.shape),
        'test': list(x_test.shape)
    }
    data_info['num_classes'] = int(num_classes)
    data_info['samples_per_class'] = counts.tolist()

    # Save data information
    with open(os.path.join(run_dir, 'data_info.json'), 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"\n=== MODEL ARCHITECTURE ===")
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

    # Save model architecture
    model_config = {
        'architecture': 'resnettssd',
        'frame_size': frame_size,
        'input_shape': [frame_size, x_train.shape[-1]],
        'num_classes': int(num_classes),
        'filters': 16 * con,
        'kernel_size': ks,
        'total_params': resnettssd.count_params()
    }
    with open(os.path.join(run_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"Model created with {resnettssd.count_params()} parameters")

    print(f"\n=== TRAINING CONFIGURATION ===")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)

    # Add model checkpointing
    checkpoint_path = os.path.join(run_dir, 'best_model.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # Add CSV logging
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(run_dir, 'training_log.csv'),
        append=True
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_rate=0.95, decay_steps=1000000
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    training_config = {
        'optimizer': 'Adam',
        'initial_learning_rate': 0.001,
        'decay_rate': 0.95,
        'decay_steps': 1000000,
        'loss': 'sparse_categorical_crossentropy',
        'batch_size': 8,
        'epochs': 100,
        'early_stopping_patience': 5,
        'early_stopping_monitor': 'val_accuracy'
    }
    with open(os.path.join(run_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)

    print("Training configuration saved")
    print("Starting training...")

    print(f"\n=== TRAINING PHASE ===")
    history = resnettssd.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=[callback, checkpoint_callback, csv_logger],
        batch_size=8,
        verbose=1
    )

    # Save training history
    history_data = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history_data, f, indent=2)

    # Plot and save training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n=== EVALUATION PHASE ===")
    print("Evaluating on MASKED test data (robustness test)")

    results = resnettssd.evaluate(x_test, y_test, verbose=1)
    test_acc = results[1]
    test_loss = results[0]
    print("test acc:", test_acc)
    print("test loss:", test_loss)

    y_pred_probs = resnettssd.predict(x_test)
    kappa_score = compute_cohen_kappa(y_test, y_pred_probs, num_classes)
    print('kappa score: ', kappa_score)

    # Save evaluation results
    evaluation_results = {
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'kappa_score': float(kappa_score),
        'num_test_samples': int(len(y_test)),
        'masking_applied_to_test': True,
        'masking_applied_to_train': False,
        'masking_applied_to_val': False
    }
    with open(os.path.join(run_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    # Save predictions for further analysis
    np.save(os.path.join(run_dir, 'y_test_true.npy'), y_test)
    np.save(os.path.join(run_dir, 'y_test_pred_probs.npy'), y_pred_probs)

    print(f"\n=== GENERATING VISUALIZATIONS ===")
    # Generate and save embeddings visualization
    try:
        visualize_embeddings(resnettssd, x_test, y_test, layer_name='dense',
                             save_path=os.path.join(graphs_dir, f'embeddings_tsne_{timestamp}.png'))
    except Exception as e:
        print(f"Warning: Could not generate embeddings visualization: {e}")

    # Generate and save ROC curves
    try:
        plot_roc_curves(resnettssd, x_test, y_test,
                        save_path=os.path.join(graphs_dir, f'roc_curves_{timestamp}.png'))
    except Exception as e:
        print(f"Warning: Could not generate ROC curves: {e}")

    # Save final model (in addition to best checkpoint)
    final_model_path = os.path.join(run_dir, 'final_model.h5')
    resnettssd.save(final_model_path)

    print(f"\n=== RUN COMPLETED ===")
    print(f"Run timestamp: {timestamp}")
    print(f"Results saved in: {run_dir}")
    print(f"Graphs saved in: {graphs_dir}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Kappa score: {kappa_score:.4f}")
    print("=" * 50)

    return test_acc, kappa_score


def visualize_embeddings(model, x_test, y_test, layer_name='dense', save_path=None):
    """
    Visualize learned embeddings using t-SNE without sklearn.
    """
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = feature_extractor.predict(x_test)

    print("Applying t-SNE...")
    tsne_results = compute_tsne(features, n_components=2, random_state=42)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y_test)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = y_test == label
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], c=[colors[i]], label=f'User {label}', alpha=0.7)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Learned Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embeddings visualization saved to: {save_path}")
    plt.show()


def plot_roc_curves(model, x_test, y_test, save_path=None):
    """
    Plot ROC curves for each class without sklearn.
    """
    y_pred_proba = model.predict(x_test)
    n_classes = y_pred_proba.shape[1]

    plt.figure(figsize=(12, 8))

    for i in range(n_classes):
        y_true_binary = (y_test == i).astype(int)
        y_scores = y_pred_proba[:, i]
        fpr, tpr, _ = compute_roc_curve(y_true_binary, y_scores)
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    plt.show()