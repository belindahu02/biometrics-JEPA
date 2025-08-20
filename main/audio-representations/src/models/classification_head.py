import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any, Dict, Optional
from torchmetrics import Accuracy, F1Score


class JEPAClassifier(L.LightningModule):
    """
    Classification head on top of pre-trained JEPA model for user identification.
    """

    def __init__(
            self,
            jepa_model: nn.Module,
            num_users: int = 109,
            hidden_dim: int = 256,
            dropout_rate: float = 0.3,
            freeze_jepa: bool = True,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['jepa_model'])

        # Store the pre-trained JEPA model
        self.jepa_model = jepa_model

        # Freeze JEPA weights if specified
        if freeze_jepa:
            for param in self.jepa_model.parameters():
                param.requires_grad = False

        # Get the output dimension from JEPA model
        # You may need to adjust this based on your JEPA model's architecture
        jepa_output_dim = self._get_jepa_output_dim()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(jepa_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_users)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_users)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_users)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_users)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_users, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_users, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_users, average="macro")

    def _get_jepa_output_dim(self) -> int:
        """
        Determine the output dimension of the JEPA model.
        You'll need to adapt this based on your specific JEPA architecture.
        """
        # Option 1: If you know the dimension
        # return 512  # Replace with actual dimension

        # Option 2: Infer from a dummy forward pass
        dummy_input = torch.randn(1, 8, 1000)  # Adjust shape based on your input
        with torch.no_grad():
            dummy_output = self.jepa_model(dummy_input)
            # If output is a dictionary or tuple, extract the relevant tensor
            if isinstance(dummy_output, dict):
                # Adjust key based on your JEPA model's output structure
                dummy_output = dummy_output['representations']  # or whatever key
            elif isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]  # or whichever element

            return dummy_output.shape[-1]  # Assuming last dim is feature dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through JEPA + classifier.

        Args:
            x: Input tensor of shape (batch_size, 8, sequence_length)
               representing 8 EEG channels

        Returns:
            User predictions of shape (batch_size, num_users)
        """
        # Get representations from JEPA model
        with torch.set_grad_enabled(not self.hparams.freeze_jepa):
            jepa_output = self.jepa_model(x)

            # Extract the relevant representation
            if isinstance(jepa_output, dict):
                representations = jepa_output['representations']  # Adjust key
            elif isinstance(jepa_output, tuple):
                representations = jepa_output[0]  # Adjust index
            else:
                representations = jepa_output

        # If representations are per-channel, we might need to aggregate
        # This depends on your JEPA model's output structure
        if representations.dim() > 2:
            # Option 1: Global average pooling
            representations = representations.mean(dim=1)  # Average across channels/time
            # Option 2: Take CLS token if available
            # representations = representations[:, 0]  # First token as CLS
            # Option 3: Max pooling
            # representations = representations.max(dim=1)[0]

        # Pass through classifier
        logits = self.classifier(representations)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)

        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Only optimize classifier parameters if JEPA is frozen
        if self.hparams.freeze_jepa:
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            # Fine-tune entire model with different learning rates
            optimizer = torch.optim.AdamW([
                {'params': self.jepa_model.parameters(), 'lr': self.hparams.learning_rate * 0.1},
                {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate}
            ], weight_decay=self.hparams.weight_decay)

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            }
        }

    def on_train_start(self):
        """Log model info at start of training."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.logger.experiment.add_text(
            "model_info",
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"JEPA frozen: {self.hparams.freeze_jepa}"
        )