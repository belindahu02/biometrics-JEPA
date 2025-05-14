import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

# === dataset/eeg_dataset.py ===
class EEGDataset(Dataset):
    def __init__(self, data, labels, mask_ratio=0.3):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()  # (T, C)
        mask = torch.ones_like(x)
        num_mask = int(self.mask_ratio * x.shape[0])
        mask_indices = random.sample(range(x.shape[0]), num_mask)
        x[mask_indices] = 0
        mask[mask_indices] = 0

        return {
            'input': x,         # masked input
            'target': self.data[idx],  # original unmasked
            'mask': mask,       # mask indicating visible/hidden frames
            'label': self.labels[idx]
        }

# === models/jepa.py ===
class JEPA(nn.Module):
    def __init__(self, input_dim=24, time_steps=30, emb_dim=128):
        super(JEPA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projector = nn.Linear(128, emb_dim)
        self.reconstructor = nn.Linear(emb_dim, input_dim * time_steps)

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        z = self.encoder(x).squeeze(-1)  # (B, 128)
        z = self.projector(z)  # (B, emb_dim)
        recon = self.reconstructor(z)  # (B, T*C)
        return recon.view(x.shape[0], -1, x.shape[2])  # (B, T, C)

# === eeg_loader.py ===
from your_preprocessing_script import data_load, norma  # Replace with actual script

def load_eeg_data(data_path, batch_size=64, mask_ratio=0.3):
    x_train, y_train = data_load(data_path)
    x_train, _ = norma(x_train)
    dataset = EEGDataset(x_train, y_train, mask_ratio=mask_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

# === train.py ===
from models.jepa import JEPA
from eeg_loader import load_eeg_data

def semantic_gap_loss(pred, target, mask):
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    loss = loss * (1 - mask)  # focus on masked regions
    return loss.sum() / (1 - mask).sum()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = load_eeg_data('path/to/data', batch_size=64, mask_ratio=0.3)
model = JEPA(input_dim=24, time_steps=30, emb_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs = batch['input'].to(device)       # masked input
        targets = batch['target'].to(device)     # ground truth
        masks = batch['mask'].to(device)         # binary mask

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = semantic_gap_loss(outputs, targets, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
