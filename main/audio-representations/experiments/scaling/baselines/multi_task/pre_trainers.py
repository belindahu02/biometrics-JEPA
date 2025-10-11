import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
import gc

from backbones import *
from data_loader import *


def pre_trainer(scen):
    """Pre-train feature extractor"""
    frame_size = 40
    path = "/app/data/1.0.0"
    # path = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0"  # Update this path

    # Use all users for pre-training (adjust as needed)
    users = list(range(1, 110))
    train_sessions = list(range(1, 15))  # R01-R10 for pre-training

    print(f"Loading pre-training data for {len(users)} users...")
    x_train, y_train, sessions_train = data_load_origin(
        path, users=users, sessions=train_sessions, frame_size=frame_size
    )
    print("Pre-training samples:", x_train.shape[0])

    x_train = norma_pre(x_train)
    print("x_train", x_train.shape)

    # Use all 6 transformations
    transformations = np.array([
        DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop
    ])
    sigma_l = np.array([0.1, 0.2, 0.2, None, None, 3])

    print(f"Applying {len(transformations)} augmentations...")
    x_train, y_train = aug_data(x_train, y_train, transformations, sigma_l, ext=False, batch_size=100)

    con = 3
    ks = 3

    def trunk():
        input_ = Input(shape=(frame_size, x_train.shape[-1]), name='input_')
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
        inputs.append(Input(shape=(frame_size, x_train.shape[-1]), name=name))

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
        metrics=['accuracy']
    )

    model.summary()

    class Logger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = []
            for i in range(len(transformations)):
                acc.append(logs.get('head_' + str(i + 1) + '_accuracy'))
            print('=' * 30, epoch + 1, '=' * 30)
            print('accuracy', acc)

            # Force garbage collection periodically
            if epoch % 5 == 0:
                gc.collect()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.1, patience=5, restore_best_weights=True
    )

    x_ = []
    y_ = []
    for i in range(len(transformations)):
        x_.append(x_train[i])
        y_.append(y_train[i])

    # Fixed batch size of 32
    history = model.fit(x_, y_, epochs=30, shuffle=True, callbacks=[Logger()],
                        verbose=1, batch_size=32)

    fet_extrct = model.layers[len(transformations)]

    # Clear memory before visualization
    del x_train, y_train, x_, y_
    gc.collect()

    # Visualize latent space using PCA (subset for memory)
    print("Generating latent space visualization...")
    x_train_viz, y_train_viz, _ = data_load_origin(
        path, users=users[:10], sessions=train_sessions[:2], frame_size=frame_size
    )
    x_train_viz = norma_pre(x_train_viz)

    # Sample subset to avoid memory issues during visualization only
    if x_train_viz.shape[0] > 2000:
        indices = np.random.choice(x_train_viz.shape[0], 2000, replace=False)
        x_train_viz = x_train_viz[indices]
        y_train_viz = y_train_viz[indices]

    enc_results = fet_extrct(x_train_viz)
    enc_results = np.array(enc_results)

    # Simple 2D projection using PCA
    enc_centered = enc_results - np.mean(enc_results, axis=0)
    cov_matrix = np.cov(enc_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    X_embedded = enc_centered @ eigenvectors[:, :2]

    fig4 = plt.figure(figsize=(18, 12))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train_viz, alpha=0.6)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='User ID')
    plt.savefig('graphs/latentspace_scen_' + str(scen) + '.png', dpi=150)
    plt.close(fig4)

    # Clean up
    del x_train_viz, y_train_viz, enc_results
    gc.collect()

    return fet_extrct
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
        metrics=['accuracy']
    )

    model.summary()

    class Logger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = []
            for i in range(len(transformations)):
                acc.append(logs.get('head_' + str(i + 1) + '_accuracy'))
            print('=' * 30, epoch + 1, '=' * 30)
            print('accuracy', acc)

            # Force garbage collection periodically
            if epoch % 5 == 0:
                gc.collect()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.1, patience=5, restore_best_weights=True
    )

    x_ = []
    y_ = []
    for i in range(len(transformations)):
        x_.append(x_train[i])
        y_.append(y_train[i])

    history = model.fit(x_, y_, epochs=30, shuffle=True, callbacks=[Logger()],
                        verbose=1, batch_size=32)  # Added batch_size

    fet_extrct = model.layers[len(transformations)]

    # Clear memory before visualization
    del x_train, y_train, x_, y_
    gc.collect()

    # Visualize latent space using PCA (subset for memory)
    print("Generating latent space visualization...")
    x_train_viz, y_train_viz, _ = data_load_origin(
        path, users=users[:10], sessions=train_sessions[:2], frame_size=frame_size,
        max_samples_per_session=5000  # Heavily subsample for visualization
    )
    x_train_viz = norma_pre(x_train_viz)

    # Sample subset to avoid memory issues
    if x_train_viz.shape[0] > 2000:
        indices = np.random.choice(x_train_viz.shape[0], 2000, replace=False)
        x_train_viz = x_train_viz[indices]
        y_train_viz = y_train_viz[indices]

    enc_results = fet_extrct(x_train_viz)
    enc_results = np.array(enc_results)

    # Simple 2D projection using PCA
    enc_centered = enc_results - np.mean(enc_results, axis=0)
    cov_matrix = np.cov(enc_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    X_embedded = enc_centered @ eigenvectors[:, :2]

    fig4 = plt.figure(figsize=(18, 12))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train_viz, alpha=0.6)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='User ID')
    plt.savefig('graphs/latentspace_scen_' + str(scen) + '.png', dpi=150)
    plt.close(fig4)

    # Clean up
    del x_train_viz, y_train_viz, enc_results
    gc.collect()

    return fet_extrct
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
        metrics=['accuracy']
    )

    model.summary()

    class Logger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            acc = []
            val_acc = []
            for i in range(len(transformations)):
                acc.append(logs.get('head_' + str(i + 1) + '_accuracy'))
                val_acc.append(logs.get('val_head_' + str(i + 1) + '_accuracy'))
            print('=' * 30, epoch + 1, '=' * 30)
            print('accuracy', acc)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.1, patience=5, restore_best_weights=True
    )
    x_ = []
    y_ = []
    for i in range(len(transformations)):
        x_.append(x_train[i])
        y_.append(y_train[i])

    history = model.fit(x_, y_, epochs=30, shuffle=True, callbacks=[Logger()], verbose=False)

    fet_extrct = model.layers[len(transformations)]

    # Visualize latent space using TensorFlow's PCA for dimensionality reduction
    x_train_viz, y_train_viz, _ = data_load_origin(
        path, users=users, sessions=train_sessions, frame_size=30
    )
    x_train_viz = norma_pre(x_train_viz)

    # Sample subset to avoid memory issues
    if x_train_viz.shape[0] > 5000:
        indices = np.random.choice(x_train_viz.shape[0], 5000, replace=False)
        x_train_viz = x_train_viz[indices]
        y_train_viz = y_train_viz[indices]

    enc_results = fet_extrct(x_train_viz)
    enc_results = np.array(enc_results)

    # Simple 2D projection using PCA-like approach
    # Center the data
    enc_centered = enc_results - np.mean(enc_results, axis=0)
    # Compute covariance matrix
    cov_matrix = np.cov(enc_centered.T)
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    # Project onto first 2 principal components
    X_embedded = enc_centered @ eigenvectors[:, :2]

    fig4 = plt.figure(figsize=(18, 12))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train_viz)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='User ID')
    plt.savefig('graphs/latentspace_scen_' + str(scen) + '.png')
    plt.close(fig4)

    return fet_extrct