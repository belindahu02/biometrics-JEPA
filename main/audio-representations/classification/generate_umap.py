import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import argparse


def log(msg: str):
    """Helper function for timestamped logging"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_embeddings(embeddings_dir, csv_file=None):
    """
    Load all embeddings from the directory and return as array with filenames

    Args:
        embeddings_dir: Path to directory containing .npy embedding files
        csv_file: Optional CSV file to filter which embeddings to load

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        filenames: list of corresponding filenames
    """
    embeddings_dir = Path(embeddings_dir)

    # Get list of embedding files
    embedding_files = list(embeddings_dir.glob("**/*_emb.npy"))
    log(f"Found {len(embedding_files)} embedding files")

    if len(embedding_files) == 0:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    # Load embeddings and filenames
    embeddings = []
    filenames = []

    for emb_file in embedding_files:
        try:
            emb = np.load(emb_file)
            # Handle different embedding shapes (flatten if needed)
            if emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1) if emb.shape[0] > 1 else emb.flatten()
            elif emb.ndim == 2:
                emb = emb.flatten()

            embeddings.append(emb)
            # Extract original filename from embedding filename
            original_name = emb_file.stem.replace('_emb', '') + '.npy'
            filenames.append(original_name)

        except Exception as e:
            log(f"Error loading {emb_file}: {e}")
            continue

    if len(embeddings) == 0:
        raise ValueError("No embeddings could be loaded successfully")

    embeddings = np.array(embeddings)
    log(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

    return embeddings, filenames


def create_labels_from_filenames(filenames, label_strategy='prefix'):
    """
    Create labels from filenames for coloring the UMAP plot

    Args:
        filenames: List of filenames
        label_strategy: Strategy for creating labels
            - 'prefix': Use first part of filename before first underscore
            - 'directory': Use directory structure if present
            - 'extension': Use file extension
            - 'none': No labels (all same color)

    Returns:
        labels: List of labels for each filename
    """
    if label_strategy == 'none':
        return ['data'] * len(filenames)
    elif label_strategy == 'prefix':
        labels = []
        for fname in filenames:
            # Try to extract meaningful prefix
            base = Path(fname).stem
            if '_' in base:
                prefix = base.split('_')[0]
            else:
                prefix = base[:3]  # First 3 characters
            labels.append(prefix)
        return labels
    elif label_strategy == 'directory':
        labels = []
        for fname in filenames:
            parts = Path(fname).parts
            if len(parts) > 1:
                labels.append(parts[0])  # First directory
            else:
                labels.append('root')
        return labels
    else:
        return ['data'] * len(filenames)


def standardize_embeddings(embeddings):
    """
    Manually standardize embeddings (zero mean, unit variance)

    Args:
        embeddings: numpy array of embeddings

    Returns:
        standardized embeddings
    """
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    return (embeddings - mean) / std


def plot_umap(embeddings, labels, filenames, save_path=None, title="UMAP Visualization of JEPA Embeddings"):
    """
    Create and plot UMAP visualization

    Args:
        embeddings: numpy array of embeddings
        labels: list of labels for coloring
        filenames: list of corresponding filenames
        save_path: path to save the plot
        title: title for the plot
    """
    log("Computing UMAP projection...")

    # Standardize embeddings manually
    embeddings_scaled = standardize_embeddings(embeddings)

    # Compute UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings_scaled)
    log(f"UMAP completed. Embedding shape: {embedding_2d.shape}")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Get unique labels and assign colors
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

    # Add legend if we have multiple labels
    if len(unique_labels) > 1 and len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log(f"Plot saved to {save_path}")

    plt.show()

    return embedding_2d, reducer


def create_interactive_plot(embeddings_2d, labels, filenames, save_path=None):
    """
    Create an interactive plot with hover information (requires plotly)
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Create DataFrame for plotly
        df = pd.DataFrame({
            'UMAP_1': embeddings_2d[:, 0],
            'UMAP_2': embeddings_2d[:, 1],
            'label': labels,
            'filename': filenames
        })

        fig = px.scatter(
            df,
            x='UMAP_1',
            y='UMAP_2',
            color='label',
            hover_data=['filename'],
            title='Interactive UMAP Visualization of JEPA Embeddings'
        )

        if save_path:
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            log(f"Interactive plot saved to {html_path}")

        fig.show()

    except ImportError:
        log("Plotly not available. Skipping interactive plot. Install with: pip install plotly")


def main():
    parser = argparse.ArgumentParser(description='Visualize JEPA embeddings with UMAP')
    parser.add_argument('--embeddings_dir', type=str,
                        default='/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/eval_embeddings',
                        help='Directory containing embedding files')
    parser.add_argument('--csv_file', type=str,
                        default='/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/files_audioset.csv',
                        help='Original CSV file (optional, for reference)')
    parser.add_argument('--output_dir', type=str, default='./umap_plots',
                        help='Directory to save plots')
    parser.add_argument('--label_strategy', type=str, default='prefix',
                        choices=['prefix', 'directory', 'none'],
                        help='Strategy for creating labels from filenames')
    parser.add_argument('--interactive', action='store_true',
                        help='Create interactive plot (requires plotly)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    log("Loading embeddings...")
    embeddings, filenames = load_embeddings(args.embeddings_dir, args.csv_file)

    # Create labels
    log(f"Creating labels using strategy: {args.label_strategy}")
    labels = create_labels_from_filenames(filenames, args.label_strategy)
    log(f"Created {len(set(labels))} unique labels")

    # Create UMAP plots
    log("Creating UMAP visualizations...")
    embedding_2d, reducer = plot_umap(embeddings, labels, filenames, args.output_dir)

    # Create summary plot
    create_summary_plot(embedding_2d, labels, args.output_dir)

    # Save embedding coordinates
    coords_df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'umap_1': embedding_2d[:, 0],
        'umap_2': embedding_2d[:, 1]
    })
    coords_path = os.path.join(args.output_dir, 'umap_coordinates.csv')
    coords_df.to_csv(coords_path, index=False)
    log(f"UMAP coordinates saved to {coords_path}")

    # Print some statistics
    log("=== UMAP Visualization Statistics ===")
    log(f"Total samples: {len(embeddings)}")
    log(f"Embedding dimension: {embeddings.shape[1]}")
    log(f"Unique labels: {len(set(labels))}")
    log(f"Label distribution: {dict(pd.Series(labels).value_counts())}")


if __name__ == "__main__":
    main()