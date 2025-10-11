import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from projectors import *
from predictors import *
from simsiam import *
import gc
import os

from backbones import *
from data_loader import *
from transformations import *

def tsne_manual_simple(X, n_components=2, n_iter=300, learning_rate=200.0):
    """
  Simplified t-SNE implementation for visualization
  Much faster but less accurate than full t-SNE
  """
    n_samples = X.shape[0]

    # PCA initialization for faster convergence
    X_centered = X - np.mean(X, axis=0)
    cov = np.dot(X_centered.T, X_centered) / n_samples
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    Y = np.dot(X_centered, eigenvectors[:, :n_components]) * 0.0001

    # Compute pairwise distances efficiently
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    D = np.maximum(D, 0)

    # Simplified probability computation
    P = np.exp(-D / (2 * 30.0))  # Fixed perplexity = 30
    np.fill_diagonal(P, 0)
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)

    # Gradient descent
    momentum = 0.5
    Y_velocity = np.zeros_like(Y)

    for iter in range(n_iter):
        # Compute Q
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n_samples):
            grad[i] = 4 * np.sum((PQ_diff[:, i] * num[:, i])[:, np.newaxis] * (Y[i] - Y), axis=0)

        # Update with momentum
        Y_velocity = momentum * Y_velocity - learning_rate * grad
        Y = Y + Y_velocity

        # Increase momentum after 20 iterations
        if iter == 20:
            momentum = 0.8

        if iter % 50 == 0:
            print(f"t-SNE iteration {iter}/{n_iter}")

    return Y


def pre_trainer(scen, fet, base_dir="/app/data/experiments/scaling/baselines"):
    frame_size = 40
    BATCH_SIZE = 40
    origin = False
    EPOCHS = 100
    path = "/app/data/1.0.0"
    # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"  # Update this path

    # Create necessary directories
    checkpoint_dir = os.path.join(base_dir, "simsiam/checkpoints")
    graphs_dir = os.path.join(base_dir, "simsiam/graphs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # Use smaller subset of users for pre-training to save memory
    # Can increase if memory allows
    users = list(range(1, 110))  # Use first 50 users instead of 100
    folder_train = ["TrainingSet"]

    print("Loading pre-training data...")
    # Limit samples per user during pre-training to manage memory
    max_samples_per_user = 500  # Adjust based on available memory

    x_train, y_train, sessions_train = data_load_origin(
        path, users=users, folders=folder_train,
        frame_size=frame_size, max_samples_per_user=max_samples_per_user
    )
    print("Training samples:", x_train.shape[0])
    num_sample = x_train.shape[0]

    if x_train.shape[0] == 0:
        raise ValueError("No training data loaded. Check path and user folders.")

    x_train = norma_pre(x_train)
    print("x_train normalized:", x_train.shape)

    def aug1_numpy(x):
        x = DA_Jitter(x, 0.5)
        x = DA_Scaling(x, 1)
        return x.astype(np.float32)  # ✅ ensure correct dtype

    @tf.function
    def tf_aug1(input):
        y = tf.numpy_function(aug1_numpy, [input], tf.float32)
        y.set_shape((BATCH_SIZE, input.shape[-1]))
        return y

    def aug2_numpy(x):
        x = DA_Jitter(x, 0.5)
        x = DA_Scaling(x, 1)
        return x.astype(np.float32)  # ✅ ensure correct dtype

    @tf.function
    def tf_aug2(input):
        y = tf.numpy_function(aug2_numpy, [input], tf.float32)
        y.set_shape((BATCH_SIZE, input.shape[-1]))
        return y

    AUTO = tf.data.AUTOTUNE
    SEED = 34

    ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=SEED)
            .map(tf_aug1, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=SEED)
            .map(tf_aug2, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

    mlp_s = 2048
    con = 3
    ks = 3
    num_training_samples = len(x_train)
    steps = EPOCHS * (num_training_samples // BATCH_SIZE)
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5, decay_steps=steps
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True, min_delta=0.0001
    )

    # Model checkpoint to save during training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f"simsiam_pretrain_scen{scen}_epoch{{epoch:02d}}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch',
        verbose=0
    )

    en = get_encoder(frame_size, x_train.shape[-1], mlp_s, origin)
    en.summary()

    contrastive = Contrastive(
        get_encoder(frame_size, x_train.shape[-1], mlp_s, origin),
        get_predictor(mlp_s, origin)
    )
    contrastive.compile(optimizer=tf.keras.optimizers.Adam(lr_decayed_fn))

    print("Training SimSiam model...")
    history = contrastive.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping, checkpoint_callback])

    backbone = tf.keras.Model(
        contrastive.encoder.input, contrastive.encoder.output
    )

    backbone.summary()

    backbone = tf.keras.Model(backbone.input, backbone.layers[-fet].output)

    backbone.summary()

    # Visualize latent space with limited samples
    print("Generating latent space visualization...")
    x_train_viz, y_train_viz, _ = data_load_origin(
        path, users=users[:10], folders=folder_train,
        frame_size=frame_size, max_samples_per_user=200
    )

    if x_train_viz.shape[0] > 0:
        x_train_viz = norma_pre(x_train_viz)

        # Limit to 2000 samples for visualization
        max_viz_samples = 2000
        if x_train_viz.shape[0] > max_viz_samples:
            indices = np.random.choice(x_train_viz.shape[0], max_viz_samples, replace=False)
            x_train_viz = x_train_viz[indices]
            y_train_viz = y_train_viz[indices]

        print(f"Computing embeddings for {x_train_viz.shape[0]} samples...")

        # Compute embeddings in batches
        batch_size = 64
        enc_results = []
        for i in range(0, len(x_train_viz), batch_size):
            batch = x_train_viz[i:i + batch_size]
            batch_enc = backbone(batch, training=False)
            enc_results.append(batch_enc.numpy())
            del batch, batch_enc
            gc.collect()

        enc_results = np.concatenate(enc_results, axis=0)

        print("Computing t-SNE embedding...")
        X_embedded = tsne_manual_simple(enc_results, n_components=2, n_iter=300)

        fig4 = plt.figure(figsize=(18, 12))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                              c=y_train_viz, cmap='tab20', alpha=0.6, s=20)
        plt.colorbar(scatter, label='User ID')
        plt.title('Latent Space Visualization (t-SNE)', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.tight_layout()

        latentspace_path = os.path.join(graphs_dir, 'latentspace_scen_1.png')
        plt.savefig(latentspace_path, dpi=150, bbox_inches='tight')
        print("Latent space visualization saved: graphs/latentspace_scen_1.png")
        plt.close(fig4)

        # Clean up
        del x_train_viz, y_train_viz, enc_results, X_embedded
        gc.collect()

    # Clean up pre-training data
    del x_train, y_train, contrastive
    tf.keras.backend.clear_session()
    gc.collect()

    print("Pre-training completed!")

    return backbone