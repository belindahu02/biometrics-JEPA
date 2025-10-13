import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
import gc

from backbones import *
from data_loader import *

base_dir = "/app/data/experiments/scaling/baselines"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "multi_task/graph_data")
graphs_dir = os.path.join(base_dir, "multi_task/graphs")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

class AugmentedDataSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for on-the-fly data augmentation with proper batch indexing
    """

    def __init__(self, base_generator, transformations, sigma_l, ext=False,
                 batches_per_epoch=100):
        self.base_generator = base_generator
        self.transformations = transformations
        self.sigma_l = sigma_l
        self.ext = ext
        self.n_transforms = len(transformations)
        self.batches_per_epoch = batches_per_epoch

        # Cache batches for the epoch
        self.cached_batches = []
        self.cache_epoch_data()

    def cache_epoch_data(self):
        """Cache data batches for the epoch"""
        self.cached_batches = []
        batch_count = 0

        for batch_x, batch_y in self.base_generator:
            self.cached_batches.append((batch_x, batch_y))
            batch_count += 1
            if batch_count >= self.batches_per_epoch:
                break

    def __len__(self):
        # Each cached batch generates 2 batches per transformation
        # (one for original, one for augmented)
        return len(self.cached_batches) * self.n_transforms * 2

    def __getitem__(self, idx):
        batches_per_transform = len(self.cached_batches) * 2
        transform_idx = idx // batches_per_transform
        within_transform_idx = idx % batches_per_transform

        base_batch_idx = within_transform_idx // 2
        is_augmented = within_transform_idx % 2 == 1

        # Get base batch
        batch_x, batch_y = self.cached_batches[base_batch_idx]

        # Prepare input arrays per head
        x_list = [np.array(batch_x, dtype=np.float32) for _ in range(self.n_transforms)]

        # Prepare output arrays per head
        y_list = [np.zeros(len(batch_x), dtype=np.float32) for _ in range(self.n_transforms)]

        if is_augmented:
            transform = self.transformations[transform_idx]
            sigma = self.sigma_l[transform_idx]
            augmented = np.array([transform(x, sigma=sigma) for x in batch_x], dtype=np.float32)
            x_list[transform_idx] = augmented
            y_list[transform_idx] = np.ones(len(batch_x), dtype=np.float32)

        # Convert lists to dictionaries with layer names
        x_dict = {f"input_{i + 1}": x_list[i] for i in range(self.n_transforms)}
        y_dict = {f"head_{i + 1}": y_list[i] for i in range(self.n_transforms)}

        return x_dict, y_dict

    def on_epoch_end(self):
        """Called at the end of each epoch"""
        # Re-cache data for next epoch (with new shuffle)
        self.cache_epoch_data()
        gc.collect()


def pre_trainer(scen):
    """Pre-train feature extractor using generators"""
    frame_size = 40
    path = "/app/data/1.0.0"
    # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"  # Update this path

    batch_size = 8  # Small batch size to manage memory

    # Use all users for pre-training
    users = list(range(1, 109))
    train_sessions = list(range(1, 11))  # R01-R10 for pre-training

    print(f"Setting up pre-training for {len(users)} users...")

    # Compute normalization statistics first
    print("Computing normalization statistics...")
    mean, std = compute_normalization_stats(
        path, users=users, sessions=train_sessions,
        frame_size=frame_size, max_samples_per_session=10000
    )

    # Create base generator
    print("Creating data generator...")
    base_generator = EEGDataGenerator(
        path, users, train_sessions, frame_size,
        batch_size=batch_size,
        max_samples_per_session=10000,  # Limit to manage memory
        mean=mean, std=std, shuffle=True
    )

    steps_per_epoch = base_generator.get_steps_per_epoch()
    print(f"Estimated steps per epoch: {steps_per_epoch}")

    # Use a reasonable number of batches per epoch
    batches_per_epoch = min(steps_per_epoch, 500)  # Cap at 500 to avoid too long epochs
    print(f"Using {batches_per_epoch} batches per epoch")

    # Get data shape from first batch
    first_batch_x, first_batch_y = next(iter(base_generator))
    n_channels = first_batch_x.shape[-1]
    print(f"Data shape: (batch_size, {frame_size}, {n_channels})")

    # Use all 6 transformations
    transformations = [
        DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop
    ]
    sigma_l = [0.1, 0.2, 0.2, None, None, 3]

    # Build model
    con = 3
    ks = 3

    def trunk():
        input_ = Input(shape=(frame_size, n_channels), name='input_')
        x = Conv1D(filters=16 * con, kernel_size=ks, strides=1, padding='same')(input_)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=4, strides=4)(x)
        x = Dropout(rate=0.1)(x)
        x = resnetblock_final(x, CR=32 * con, KS=ks)
        return tf.keras.models.Model(input_, x, name='trunk_')

    inputs = []
    for i in range(len(transformations)):
        name = 'input_' + str(i + 1)
        inputs.append(Input(shape=(frame_size, n_channels), name=name))

    trunk = trunk()
    trunk.summary()

    fets = []
    for input_ in inputs:
        fets.append(trunk(input_))

    heads = []
    for i, fet in enumerate(fets):
        dens_name = 'dens_' + str(i + 1)
        densi_name = 'densi_' + str(i + 1)
        head_name = 'head_' + str(i + 1)
        dens = Dense(256, activation='relu', name=dens_name)(fet)
        dens = Dense(64, activation='relu', name=densi_name)(dens)
        head = Dense(1, activation='sigmoid', name=head_name)(dens)
        heads.append(head)

    model = tf.keras.models.Model(inputs, heads, name='multi-task_self-supervised')

    loss = []
    loss_weights = []
    for i in range(len(transformations)):
        loss.append('binary_crossentropy')
        loss_weights.append(1 / len(transformations))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss=loss,
        loss_weights=loss_weights,
        optimizer=opt,
        metrics=['accuracy'] * len(transformations)  # one per output
    )

    model.summary()

    class Logger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = []
            for i in range(len(transformations)):
                acc.append(logs.get('head_' + str(i + 1) + '_accuracy'))
            print('=' * 30, epoch + 1, '=' * 30)
            print('accuracy', acc)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.1, patience=5, restore_best_weights=True
    )

    # Create augmented data sequence
    print("\nPreparing augmented data sequence...")
    aug_sequence = AugmentedDataSequence(
        base_generator, transformations, sigma_l,
        ext=False, batches_per_epoch=batches_per_epoch
    )

    print(f"Total batches per epoch: {len(aug_sequence)}")

    # Training with Sequence
    print("\nStarting training...")
    history = model.fit(
        aug_sequence,
        epochs=30,
        callbacks=[Logger(), callback],
        verbose=1
    )

    fet_extrct = model.layers[len(transformations)]

    # Clean up
    gc.collect()

    # Visualize latent space using subset of data
    print("\nGenerating latent space visualization...")
    viz_generator = EEGDataGenerator(
        path, users[:10], train_sessions[:2], frame_size,
        batch_size=32,
        max_samples_per_session=5000,
        mean=mean, std=std, shuffle=False
    )

    # Collect subset of data for visualization
    x_train_viz = []
    y_train_viz = []
    samples_collected = 0
    max_samples = 2000

    for batch_x, batch_y in viz_generator:
        x_train_viz.append(batch_x)
        y_train_viz.append(batch_y)
        samples_collected += len(batch_x)
        if samples_collected >= max_samples:
            break

    x_train_viz = np.concatenate(x_train_viz, axis=0)[:max_samples]
    y_train_viz = np.concatenate(y_train_viz, axis=0)[:max_samples]

    # Extract features
    enc_results = fet_extrct.predict(x_train_viz, batch_size=32, verbose=0)
    enc_results = np.array(enc_results)

    # Flatten if needed
    if len(enc_results.shape) > 2:
        enc_results = enc_results.reshape(enc_results.shape[0], -1)

    # Simple 2D projection using PCA
    enc_centered = enc_results - np.mean(enc_results, axis=0)
    cov_matrix = np.cov(enc_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    X_embedded = enc_centered @ eigenvectors[:, :2].real

    fig4 = plt.figure(figsize=(18, 12))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train_viz, alpha=0.6, cmap='tab10')
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='User ID')
    plt.savefig(os.path.join(graphs_dir, 'latentspace_scen_' + str(scen) + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # Clean up
    del x_train_viz, y_train_viz, enc_results
    gc.collect()

    print("\nPre-training complete!")
    return fet_extrct