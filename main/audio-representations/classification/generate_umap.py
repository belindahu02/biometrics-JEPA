import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import umap
from datetime import datetime

# =============================
# Hardcoded paths and settings
# =============================
EMBEDDINGS_DIR = '/app/data/embeddings_subset10'
OUTPUT_DIR = '/app/data/umap_subset10'
LABEL_STRATEGY = 'prefix'  # options: 'prefix', 'directory', 'none'


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_embeddings(embeddings_dir):
    embeddings_dir = Path(embeddings_dir)
    embedding_files = list(embeddings_dir.glob("**/*_emb.npy"))
    log(f"Found {len(embedding_files)} embedding files")

    if len(embedding_files) == 0:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    embeddings, filenames = [], []

    for emb_file in embedding_files:
        try:
            emb = np.load(emb_file)
            if emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1) if emb.shape[0] > 1 else emb.flatten()
            elif emb.ndim == 2:
                emb = emb.flatten()

            embeddings.append(emb)
            filenames.append(emb_file.stem.replace('_emb', '') + '.npy')

        except Exception as e:
            log(f"Error loading {emb_file}: {e}")
            continue

    if len(embeddings) == 0:
        raise ValueError("No embeddings could be loaded successfully")

    embeddings = np.array(embeddings)
    log(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

    return embeddings, filenames


def create_labels_from_filenames(filenames, label_strategy='prefix'):
    if label_strategy == 'none':
        return ['data'] * len(filenames)
    elif label_strategy == 'prefix':
        labels = []
        for fname in filenames:
            base = Path(fname).stem
            if '_' in base:
                prefix = base.split('_')[0]
            else:
                prefix = base[:3]
            labels.append(prefix)
        return labels
    elif label_strategy == 'directory':
        labels = []
        for fname in filenames:
            parts = Path(fname).parts
            labels.append(parts[0] if len(parts) > 1 else 'root')
        return labels
    else:
        return ['data'] * len(filenames)


def standardize_embeddings(embeddings):
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    std = np.where(std == 0, 1, std)
    return (embeddings - mean) / std


def plot_umap(embeddings, labels, filenames, save_path=None, title="UMAP Visualization of JEPA Embeddings"):
    log("Computing UMAP projection...")
    embeddings_scaled = standardize_embeddings(embeddings)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings_scaled)
    log(f"UMAP completed. Embedding shape: {embedding_2d.shape}")

    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=50
        )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title)

    if 1 < len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log(f"Plot saved to {save_path}")

    plt.close()  # Close plot to avoid display issues in SSH
    return embedding_2d, reducer


# =============================
# Main script
# =============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

log("Loading embeddings...")
embeddings, filenames = load_embeddings(EMBEDDINGS_DIR)

log(f"Creating labels using strategy: {LABEL_STRATEGY}")
labels = create_labels_from_filenames(filenames, LABEL_STRATEGY)
log(f"Created {len(set(labels))} unique labels")

save_path = os.path.join(OUTPUT_DIR, 'jepa_embeddings_umap.png')
embedding_2d, reducer = plot_umap(embeddings, labels, filenames, save_path)

coords_df = pd.DataFrame({
    'filename': filenames,
    'label': labels,
    'umap_1': embedding_2d[:, 0],
    'umap_2': embedding_2d[:, 1]
})
coords_path = os.path.join(OUTPUT_DIR, 'umap_coordinates.csv')
coords_df.to_csv(coords_path, index=False)
log(f"UMAP coordinates saved to {coords_path}")

log("=== UMAP Visualization Statistics ===")
log(f"Total samples: {len(embeddings)}")
log(f"Embedding dimension: {embeddings.shape[1]}")
log(f"Unique labels: {len(set(labels))}")
log(f"Label distribution: {dict(pd.Series(labels).value_counts())}")
