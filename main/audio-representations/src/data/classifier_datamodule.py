from typing import List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from src.data.components.classifier_dataset import build_classification_dataset


class EEGClassificationDataModule(LightningDataModule):
    """
    DataModule for EEG user classification that works with existing config structure.

    This module extends the original LMS structure to provide supervised learning
    with user labels while maintaining compatibility with existing configs.
    """

    def __init__(self,
                 data_path: str,
                 dataset: str,
                 crop_frames: int,
                 selected_channels: List[int] = [3, 12, 13, 18, 50, 60, 61, 64],
                 norm_stats: Tuple[float, float] | None = None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 devices: int | List[int] = 1,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 random_state: int = 42,
                 min_samples_per_user: int = 10):
        super().__init__()

        if not isinstance(devices, int):
            devices = len(devices)

        self.save_hyperparameters()

        # Store parameters
        self.dataset_kwargs = dict(
            data_path=data_path,
            dataset=dataset,
            crop_frames=crop_frames,
            selected_channels=selected_channels,
            norm_stats=norm_stats,
            min_samples_per_user=min_samples_per_user
        )

        self.dataloader_kwargs = dict(
            batch_size=batch_size // devices,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_users = None

        print(f"EEG Classification DataModule initialized")
        print(f"Selected channels: {selected_channels}")
        print(f"Batch size per device: {batch_size // devices}")

    def _split_users(self, all_users: List):
        """Split users into train/val/test using PyTorch's generator for reproducibility."""
        # Set random seed for reproducible splits
        generator = torch.Generator().manual_seed(self.hparams.random_state)

        # Shuffle users
        indices = torch.randperm(len(all_users), generator=generator).tolist()
        shuffled_users = [all_users[i] for i in indices]

        # Calculate split sizes
        n_users = len(all_users)
        train_size = int(self.hparams.train_split * n_users)
        val_size = int(self.hparams.val_split * n_users)

        # Split users
        train_users = shuffled_users[:train_size]
        val_users = shuffled_users[train_size:train_size + val_size]
        test_users = shuffled_users[train_size + val_size:]

        return train_users, val_users, test_users

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""

        if stage == "fit" or stage is None:
            # First, build a full dataset to get all users
            full_dataset = build_classification_dataset(
                mode='train',  # Use train mode for data loading
                **self.dataset_kwargs
            )

            self.num_users = full_dataset.num_users
            all_users = full_dataset.users

            print(f"Found {self.num_users} users: {all_users}")

            # Split users into train/val/test
            train_users, val_users, test_users = self._split_users(all_users)

            # Store test users for later use
            self._test_users = test_users

            print(f"User split - Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")

            # Create train dataset
            self.train_dataset = build_classification_dataset(
                mode='train',
                allowed_users=train_users,
                **self.dataset_kwargs
            )

            # Create validation dataset
            self.val_dataset = build_classification_dataset(
                mode='val',
                allowed_users=val_users,
                **self.dataset_kwargs
            )

            print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            # Use stored test users or recompute split
            if not hasattr(self, '_test_users'):
                # Recompute the split to get test users
                full_dataset = build_classification_dataset(
                    mode='test',
                    **self.dataset_kwargs
                )
                all_users = full_dataset.users

                train_users, val_users, test_users = self._split_users(all_users)
                self._test_users = test_users

            # Create test dataset
            self.test_dataset = build_classification_dataset(
                mode='test',
                allowed_users=self._test_users,
                **self.dataset_kwargs
            )

            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,  # Ensure consistent batch sizes
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )

    def get_num_users(self) -> int:
        """Get the number of unique users in the dataset."""
        return self.num_users if self.num_users is not None else 0

    def get_selected_channels(self) -> List[int]:
        """Get the list of selected channel indices."""
        return self.hparams.selected_channels

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'users={self.get_num_users()}, '
                f'channels={len(self.get_selected_channels())}, '
                f'batch_size={self.hparams.batch_size})')