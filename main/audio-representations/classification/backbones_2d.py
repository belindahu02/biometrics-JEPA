import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic ResNet block for 2D spectrograms
class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), skip=True):
        super().__init__()
        self.skip = skip
        self.stride = stride

        # Three convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding='same' if stride == (1, 1) else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=(1, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=(1, 1), padding='same')

        # Skip connection adjustment for channel/spatial dimension changes
        if skip and (in_channels != out_channels or stride != (1, 1)):
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward through conv layers
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        # Skip connection
        if self.skip:
            out = out + self.shortcut(x)

        out = self.relu(out)
        out = self.pool(out)
        return out


# Final block with global pooling for 2D spectrograms
class ResNetBlockFinal2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), skip=True):
        super().__init__()
        self.skip = skip

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')

        if skip and in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        # Global pooling across both time and frequency dimensions
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        if self.skip:
            out = out + self.shortcut(x)

        out = self.relu(out)
        out = self.global_pool(out)
        return out.squeeze(-1).squeeze(-1)  # [B, C]


# Complete 2D ResNet for spectrograms
class SpectrogramResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, channels=[64, 128, 256, 512]):
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, channels[0],
                                      kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.initial_bn = nn.BatchNorm2d(channels[0])
        self.initial_pool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)

        # ResNet blocks
        self.block1 = ResNetBlock2D(channels[0], channels[0])
        self.block2 = ResNetBlock2D(channels[0], channels[1], stride=(2, 2))
        self.block3 = ResNetBlock2D(channels[1], channels[2], stride=(2, 2))
        self.block4 = ResNetBlockFinal2D(channels[2], channels[3])

        # Classification head
        self.classifier = nn.Linear(channels[3], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [B, C, H, W] where H=time, W=frequency
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.classifier(x)
        return x


# Lightweight version for smaller spectrograms
class LightweightSpectrogramResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, channels=[32, 64, 128]):
        super().__init__()

        # Initial convolution (smaller kernel, less aggressive stride)
        self.initial_conv = nn.Conv2d(input_channels, channels[0],
                                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.initial_bn = nn.BatchNorm2d(channels[0])

        # ResNet blocks
        self.block1 = ResNetBlock2D(channels[0], channels[0])
        self.block2 = ResNetBlock2D(channels[0], channels[1])
        self.block3 = ResNetBlockFinal2D(channels[1], channels[2])

        # Classification head
        self.classifier = nn.Linear(channels[2], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.initial_bn(self.initial_conv(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.classifier(x)
        return x