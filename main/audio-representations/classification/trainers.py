# =============================================
# trainers_pytorch.py
# =============================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from backbones import ResNetBlockFinal
from data_loader import load_spectrogram_data_as_1d, norma, user_data_split

# --------------------------
# Training function
# --------------------------

def spectrogram_trainer(samples_per_user, data_path, user_ids, conversion_method='pca',
                        batch_size=8, epochs=100, lr=0.001, device=None):
    """
    PyTorch version of spectrogram trainer.

    Args:
        samples_per_user: Number of samples per user for training
        data_path: Path to spectrogram data
        user_ids: List of user IDs to include
        conversion_method: Method for converting spectrograms to 1D
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Initial learning rate
        device: torch device ('cuda' or 'cpu')
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    frame_size = 520
    target_features = 24

    # --------------------------
    # Load and preprocess data
    # --------------------------
    x_train, y_train, x_val, y_val, x_test, y_test, sessions = load_spectrogram_data_as_1d(
        data_path, user_ids, conversion_method=conversion_method, target_features=target_features
    )

    x_train, x_val, x_test = norma(x_train, x_val, x_test)
    x_train, y_train = user_data_split(x_train, y_train, samples_per_user=samples_per_user)

    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Move channels to PyTorch format: (batch, channels, seq_len)
    x_train = x_train.permute(0, 2, 1)
    x_val = x_val.permute(0, 2, 1)
    x_test = x_test.permute(0, 2, 1)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

    num_classes = len(np.unique(y_train.numpy()))

    # --------------------------
    # Build model
    # --------------------------
    class SpectrogramResNet(nn.Module):
        def __init__(self, input_channels, num_classes, ks=3, con=3):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 16*con, kernel_size=ks, padding='same')
            self.bn1 = nn.BatchNorm1d(16*con)
            self.resblock = ResNetBlockFinal(16 * con, out_channels=32 * con, kernel_size=ks)
            self.fc1 = nn.Linear(32 * con, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = self.resblock(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

    model = SpectrogramResNet(input_channels=target_features, num_classes=num_classes).to(device)
    print(model)

    # --------------------------
    # Training setup
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)
        val_acc = correct_val / total_val

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # --------------------------
    # Load best model
    # --------------------------
    model.load_state_dict(best_model_state)

    # --------------------------
    # Evaluation on test set
    # --------------------------
    model.eval()
    correct_test, total_test = 0, 0
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            correct_test += (preds == yb).sum().item()
            total_test += yb.size(0)
    test_acc = correct_test / total_test
    print(f"Test Accuracy: {test_acc:.4f}")

    # --------------------------
    # Compute Cohen's Kappa
    # --------------------------
    all_preds = np.concatenate(all_preds)
    y_true = y_test.numpy()
    po = np.mean(all_preds == y_true)
    pe = np.sum(np.bincount(y_true) * np.bincount(all_preds)) / (len(y_true)**2)
    kappa_score = (po - pe) / (1 - pe)
    print(f"Cohen's Kappa: {kappa_score:.4f}")

    return test_acc, kappa_score
