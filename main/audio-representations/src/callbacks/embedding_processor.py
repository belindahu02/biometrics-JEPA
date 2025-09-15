import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hydra
from omegaconf import DictConfig
from collections import defaultdict
import torch.nn.functional as F
from datetime import datetime
from typing import Optional, Union


# ---------------------------------------------------------------------------- #
# Helper function for timestamped logging
# ---------------------------------------------------------------------------- #
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- #
# Custom Dataset for filenames in CSV using complete_audio logic
# ---------------------------------------------------------------------------- #
class EvalDataset(Dataset):
    def __init__(self, csv_file, data_dir, crop_frames=208, repeat_short=True):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.crop_frames = crop_frames
        self.repeat_short = repeat_short

    def __len__(self):
        return len(self.df)

    def complete_audio(self, lms):
        l = lms.shape[-1]
        # repeat if shorter than crop_frames
        if self.repeat_short and l < self.crop_frames:
            while l < self.crop_frames:
                lms = torch.cat([lms, lms], dim=-1)
                l = lms.shape[-1]

        # crop if longer than crop_frames
        if l > self.crop_frames:
            lms = lms[..., :self.crop_frames]  # take first crop_frames frames
        # pad if shorter
        elif l < self.crop_frames:
            pad_param = [0, self.crop_frames - l] + [0, 0] * (lms.ndim - 1)
            lms = F.pad(lms, pad_param, mode='constant', value=0)

        return lms

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        data = np.load(file_path)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # add channel dim
        data = torch.tensor(data, dtype=torch.float32)
        data = self.complete_audio(data)
        return data, self.df.iloc[idx, 0]


# ---------------------------------------------------------------------------- #
# Main processing class
# ---------------------------------------------------------------------------- #
class EmbeddingProcessor:
    def __init__(self,
                 csv_file: str,
                 data_dir: str,
                 embeddings_dir: str,
                 grouped_embeddings_dir: str,
                 checkpoint_path: Optional[str] = None,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 crop_frames: Optional[int] = None):
        """
        Initialize the embedding processor.

        Args:
            csv_file: Path to CSV file containing filenames
            data_dir: Directory containing the input data files
            embeddings_dir: Directory to save individual embeddings
            grouped_embeddings_dir: Directory to save grouped/stacked embeddings
            checkpoint_path: Path to model checkpoint (optional if model is passed directly)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            crop_frames: Number of frames to crop to (will be inferred from model if None)
        """
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.embeddings_dir = embeddings_dir
        self.grouped_embeddings_dir = grouped_embeddings_dir
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_frames = crop_frames

    def precompute_embeddings(self,
                              model: Optional[torch.nn.Module] = None,
                              cfg: Optional[DictConfig] = None):
        """
        Precompute embeddings from the dataset.

        Args:
            model: Pre-instantiated model (optional)
            cfg: Hydra config (required if model is None)
        """
        log("Starting embedding precomputation...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Handle model loading
        if model is None:
            if cfg is None:
                raise ValueError("Either model or cfg must be provided")
            log("Instantiating model from config...")
            model = hydra.utils.instantiate(cfg.model)

            if self.checkpoint_path:
                log(f"Loading checkpoint from {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        model.to(device)
        log("Model ready on device.")

        # Determine crop_frames if not provided
        if self.crop_frames is None:
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'img_size'):
                self.crop_frames = model.encoder.img_size[1]
            else:
                self.crop_frames = 208  # default fallback
                log(f"Using default crop_frames: {self.crop_frames}")

        # Setup dataset
        log(f"Loading dataset from {self.csv_file}...")
        dataset = EvalDataset(
            self.csv_file,
            self.data_dir,
            crop_frames=self.crop_frames,
            repeat_short=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        log(f"Dataset loaded with {len(dataset)} samples.")

        # Create output directory
        os.makedirs(self.embeddings_dir, exist_ok=True)
        log(f"Embeddings directory ready at {self.embeddings_dir}")

        # Process batches
        with torch.no_grad():
            for i, (batch_data, filenames) in enumerate(dataloader):
                if i == 0:
                    log("Processing first batch...")

                batch_data = batch_data.to(device)
                batch_embeddings = model.encoder(batch_data)

                if isinstance(batch_embeddings, tuple):
                    batch_embeddings = batch_embeddings[0]
                batch_embeddings = batch_embeddings.cpu().numpy()

                for emb, fname in zip(batch_embeddings, filenames):
                    save_path = os.path.join(
                        self.embeddings_dir,
                        f"{os.path.splitext(fname)[0]}_emb.npy"
                    )
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, emb)

                if (i + 1) % 10 == 0:
                    log(f"Processed {i + 1} batches...")

        log(f"✅ All embeddings saved to {self.embeddings_dir}")

    def group_embeddings(self):
        """
        Group and stack embeddings by frame ID.
        """
        log("Starting embedding grouping...")

        # Walk through all leaf folders
        for root, dirs, files in os.walk(self.embeddings_dir):
            # only process leaf folders with .npy files
            npy_files = [f for f in files if f.endswith("_emb.npy")]
            if not npy_files:
                continue

            # prepare frame-wise grouping
            frames_dict = defaultdict(list)

            for f in npy_files:
                # Example filename: Af3_frame_000_emb.npy
                try:
                    frame_id = f.split("_frame_")[1].split("_")[0]
                    frames_dict[frame_id].append(f)
                except IndexError:
                    log(f"Warning: Could not parse frame ID from {f}")
                    continue

            # stack embeddings for each frame
            leaf_output_dir = os.path.join(
                self.grouped_embeddings_dir,
                os.path.relpath(root, self.embeddings_dir)
            )
            os.makedirs(leaf_output_dir, exist_ok=True)

            for frame_id, frame_files in frames_dict.items():
                frame_files.sort()
                arrays = [np.load(os.path.join(root, f)) for f in frame_files]
                stacked = np.vstack(arrays)
                save_path = os.path.join(leaf_output_dir, f"{frame_id}_stacked.npy")
                np.save(save_path, stacked)
                log(f"✅ Saved {save_path}: stacked shape {stacked.shape} from {len(frame_files)} files")

            log(f"✅ Processed {root}: {len(frames_dict)} frames stacked")

    def process_all(self,
                    model: Optional[torch.nn.Module] = None,
                    cfg: Optional[DictConfig] = None):
        """
        Run the complete pipeline: precompute embeddings and group them.

        Args:
            model: Pre-instantiated model (optional)
            cfg: Hydra config (required if model is None)
        """
        self.precompute_embeddings(model, cfg)
        self.group_embeddings()
        log("✅ Complete processing pipeline finished!")


# ---------------------------------------------------------------------------- #
# Standalone script functionality
# ---------------------------------------------------------------------------- #
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    Standalone script entry point using Hydra config.
    """
    # You can override these paths via command line or config
    processor = EmbeddingProcessor(
        csv_file=cfg.get("csv_file", "/app/data/files_evaluation_subset10.csv"),
        data_dir=cfg.get("data_dir", "/app/data"),
        embeddings_dir=cfg.get("embeddings_dir", "/app/logs/eval_embeddings"),
        grouped_embeddings_dir=cfg.get("grouped_embeddings_dir", "/app/logs/grouped_embeddings"),
        checkpoint_path=cfg.get("checkpoint_path", "/app/data/jepa_logs_subset10/xps/97d170e1/checkpoints/last.ckpt"),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 4)
    )

    processor.process_all(cfg=cfg)


if __name__ == "__main__":
    main()