import torch.nn as nn


# Basic ResNet block for 2D spectrograms WITH DROPOUT
class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), skip=True, dropout_rate=0.0):
        super().__init__()
        self.skip = skip
        self.stride = stride
        self.dropout_rate = dropout_rate

        # Three convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding='same' if stride == (1, 1) else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=(1, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=(1, 1), padding='same')

        # Add dropout after conv layers
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()

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
        out = self.dropout(out)  # Apply dropout

        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)  # Apply dropout

        out = self.conv3(out)

        # Skip connection
        if self.skip:
            out = out + self.shortcut(x)

        out = self.relu(out)
        out = self.pool(out)
        return out


# Final block with global pooling for 2D spectrograms WITH DROPOUT
class ResNetBlockFinal2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), skip=True, dropout_rate=0.0):
        super().__init__()
        self.skip = skip
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')

        # Add dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()

        if skip and in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        # Global pooling across both time and frequency dimensions
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out = self.conv3(out)

        if self.skip:
            out = out + self.shortcut(x)

        out = self.relu(out)
        out = self.global_pool(out)
        return out.squeeze(-1).squeeze(-1)  # [B, C]


# Complete 2D ResNet for spectrograms WITH DROPOUT
class SpectrogramResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, channels=None, dropout_rate=0.0, classifier_dropout=0.0):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale spectrograms)
            num_classes: Number of output classes
            channels: List of channel sizes for each block
            dropout_rate: Dropout rate for convolutional layers (Dropout2d)
            classifier_dropout: Dropout rate before final classifier (standard Dropout)
        """
        super().__init__()

        # Initial convolution
        if channels is None:
            channels = [64, 128, 256, 512]
        self.initial_conv = nn.Conv2d(input_channels, channels[0],
                                      kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.initial_bn = nn.BatchNorm2d(channels[0])
        self.initial_pool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)

        # ResNet blocks with dropout
        self.block1 = ResNetBlock2D(channels[0], channels[0], dropout_rate=dropout_rate)
        self.block2 = ResNetBlock2D(channels[0], channels[1], stride=(2, 2), dropout_rate=dropout_rate)
        self.block3 = ResNetBlock2D(channels[1], channels[2], stride=(2, 2), dropout_rate=dropout_rate)
        self.block4 = ResNetBlockFinal2D(channels[2], channels[3], dropout_rate=dropout_rate)

        # Dropout before classifier (standard dropout for 1D features)
        if classifier_dropout > 0:
            self.classifier_dropout = nn.Dropout(classifier_dropout)
        else:
            self.classifier_dropout = nn.Identity()

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

        # Apply classifier dropout
        x = self.classifier_dropout(x)

        x = self.classifier(x)
        return x


# Lightweight version for smaller spectrograms WITH DROPOUT
class LightweightSpectrogramResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, channels=None, dropout_rate=0.0, classifier_dropout=0.0):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale spectrograms)
            num_classes: Number of output classes
            channels: List of channel sizes for each block
            dropout_rate: Dropout rate for convolutional layers (Dropout2d)
            classifier_dropout: Dropout rate before final classifier (standard Dropout)
        """
        super().__init__()

        # Initial convolution (smaller kernel, less aggressive stride)
        if channels is None:
            channels = [32, 64, 128]
        self.initial_conv = nn.Conv2d(input_channels, channels[0],
                                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.initial_bn = nn.BatchNorm2d(channels[0])

        # ResNet blocks with dropout
        self.block1 = ResNetBlock2D(channels[0], channels[0], dropout_rate=dropout_rate)
        self.block2 = ResNetBlock2D(channels[0], channels[1], dropout_rate=dropout_rate)
        self.block3 = ResNetBlockFinal2D(channels[1], channels[2], dropout_rate=dropout_rate)

        # Dropout before classifier
        if classifier_dropout > 0:
            self.classifier_dropout = nn.Dropout(classifier_dropout)
        else:
            self.classifier_dropout = nn.Identity()

        # Classification head
        self.classifier = nn.Linear(channels[2], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.initial_bn(self.initial_conv(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Apply classifier dropout
        x = self.classifier_dropout(x)

        x = self.classifier(x)
        return x