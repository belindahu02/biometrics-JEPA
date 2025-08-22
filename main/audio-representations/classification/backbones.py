import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic ResNet block for 1D signals
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, skip=True):
        super().__init__()
        self.skip = skip
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')

        if skip and in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        if self.skip:
            out = out + self.shortcut(x)
        out = self.relu(out)
        out = self.pool(out)
        return out

# Final block with global pooling
class ResNetBlockFinal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, skip=True):
        super().__init__()
        self.skip = skip
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')

        if skip and in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        if self.skip:
            out = out + self.shortcut(x)
        out = self.relu(out)
        out = self.global_pool(out)
        return out.squeeze(-1)  # [B, C]
