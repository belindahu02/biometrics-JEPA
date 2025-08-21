import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hydra
from omegaconf import DictConfig

from src.utils import register_resolvers
from datetime import datetime
import torch.nn.functional as F


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
# Precompute embeddings
# ---------------------------------------------------------------------------- #
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    register_resolvers()

    log("Starting script...")

    # instantiate model
    log("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log("Model instantiated and moved to device.")

    # use mounted data path in Docker
    csv_file = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/files_audioset.csv"
    data_dir = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data"
    log(f"Loading dataset from {csv_file}...")
    dataset = EvalDataset(csv_file, data_dir, crop_frames=cfg.model.encoder.img_size[1], repeat_short=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    log(f"Dataset loaded with {len(dataset)} samples.")

    embeddings_dir = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/eval_embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    log(f"Embeddings directory ready at {embeddings_dir}")

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
                # Corrected logic to ensure directories exist
                save_path = os.path.join(embeddings_dir, f"{os.path.splitext(fname)[0]}_emb.npy")
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)

                np.save(save_path, emb)

            if (i + 1) % 10 == 0:
                log(f"Processed {i + 1} batches...")

    log(f"✅ All embeddings saved to {embeddings_dir}")


if __name__ == "__main__":
    main()