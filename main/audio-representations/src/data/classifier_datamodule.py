from typing import List, Tuple, Optional, Dict, Any
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
import numpy as np

from src.data.components.eeg_classification_dataset import EEGClassificationDataset


class EEGClassificationDataModule(LightningDataModule):
    """
    DataModule for EEG user classification task.

    This module handles loading EEG data with user labels for supervised learning.
    Unlike the unsupervised LMS module, this returns (8_channel_data, user_id) pairs.
    """

    def __init__(
            self,
            data_path: str,
            dataset: str,
            crop_frames: int,
            selected_channels: List[int] = None,  # Which 8 channels to use
            norm_stats: Tuple[float, float] | None = None,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            devices: int | List[int] = 1,
            train_split: float = 0.7,
            val_split: float = 0.15,
            test_split: float = 0.15,
            random_state: int = 42,
            min_samples_per_user: int = 10,  # Minimum samples per user to include
    ):
        super().__init__()

        if not isinstance(devices, int):
            devices = len(devices)

        # Default to first 8 channels if not specified
        if selected_channels is None:
            selected_channels = list(range(8))

        assert len(selected_channels) == 8, "Must select exactly 8 channels"
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

        self.save_hyperparameters()

        # Dataset arguments
        self.dataset_kwargs = dict(
            data_path=data_path,
            dataset_name=dataset,
            crop_frames=crop_frames,
            selected_channels=selected_channels,
            norm_stats=norm_stats,
            min_samples_per_user=min_samples_per_user
        )

        # DataLoader arguments
        self.dataloader_kwargs = dict(
            batch_size=batch_size // devices,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Store dataset info
        self.num_users = None
        self.channel_names = None

    def prepare_data(self):
        """
        Download/prepare data if needed. Called only on rank 0.
        This is where you might download data, but for EEG data
        we assume it's already available.
        """
        if not os.path.exists(self.hparams.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.hparams.data_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, validate, test, predict).
        Called on every process in DDP.
        """

        if stage == "fit" or stage is None:
            # Load full dataset
            full_dataset = EEGClassificationDataset(
                mode='full',  # Load all data for splitting
                **self.dataset_kwargs
            )

            # Get dataset info
            self.num_users = full_dataset.num_users
            self.channel_names = full_dataset.channel_names

            # Split data by users to ensure user-level separation
            user_ids = list(range(self.num_users))

            train_users, temp_users = train_test_split(
                user_ids,
                test_size=self.hparams.val_split + self.hparams.test_split,
                random_state=self.hparams.random_state,
                stratify=None  # Can't stratify with user IDs directly
            )

            val_users, test_users = train_test_split(
                temp_users,
                test_size=self.hparams.test_split / (self.hparams.val_split + self.hparams.test_split),
                random_state=self.hparams.random_state
            )

            # Create datasets with user splits
            self.train_dataset = EEGClassificationDataset(
                mode='train',
                allowed_users=train_users,
                **self.dataset_kwargs
            )

            self.val_dataset = EEGClassificationDataset(
                mode='val',
                allowed_users=val_users,
                **self.dataset_kwargs
            )

            # Store test users for later
            self._test_users = test_users

            print(f"Dataset split - Train users: {len(train_users)}, "
                  f"Val users: {len(val_users)}, Test users: {len(test_users)}")
            print(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                  f"Val: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            if not hasattr(self, '_test_users'):
                # If setup wasn't called with 'fit' first, we need to determine test users
                full_dataset = EEGClassificationDataset(mode='full', **self.dataset_kwargs)
                self.num_users = full_dataset.num_users

                # Use the same split logic to get test users
                user_ids = list(range(self.num_users))
                train_users, temp_users = train_test_split(
                    user_ids,
                    test_size=self.hparams.val_split + self.hparams.test_split,
                    random_state=self.hparams.random_state
                )
                val_users, test_users = train_test_split(
                    temp_users,
                    test_size=self.hparams.test_split / (self.hparams.val_split + self.hparams.test_split),
                    random_state=self.hparams.random_state
                )
                self._test_users = test_users

            self.test_dataset = EEGClassificationDataset(
                mode='test',
                allowed_users=self._test_users,
                **self.dataset_kwargs
            )

            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,  # Ensure consistent batch sizes
            **self.dataloader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def predict_dataloader(self) -> DataLoader:
        # Use test dataset for predictions
        return self.test_dataloader()

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return useful information about the dataset."""
        return {
            'num_users': self.num_users,
            'selected_channels': self.hparams.selected_channels,
            'channel_names': self.channel_names,
            'crop_frames': self.hparams.crop_frames,
            'train_size': len(self.train_dataset) if self.train_dataset else None,
            'val_size': len(self.val_dataset) if self.val_dataset else None,
            'test_size': len(self.test_dataset) if self.test_dataset else None,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.hparams.dataset}, "
            f"batch_size={self.hparams.batch_size}, "
            f"num_users={self.num_users}, "
            f"selected_channels={len(self.hparams.selected_channels) if self.hparams.selected_channels else 'None'})"
        )