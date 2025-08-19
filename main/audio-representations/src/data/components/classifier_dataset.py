import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset


class EEGClassificationDataset(Dataset):
    """
    Dataset for EEG user classification.

    Returns (8_channel_data, user_id) pairs for supervised learning.
    """

    def __init__(
            self,
            data_path: str,
            dataset_name: str,
            crop_frames: int,
            selected_channels: List[int],
            mode: str = 'train',  # 'train', 'val', 'test', 'full'
            allowed_users: Optional[List[int]] = None,
            norm_stats: Tuple[float, float] = None,
            min_samples_per_user: int = 10,
    ):
        """
        Args:
            data_path: Path to the data directory
            dataset_name: Name of the dataset
            crop_frames: Number of frames to crop/sample
            selected_channels: List of 8 channel indices to use
            mode: Dataset mode ('train', 'val', 'test', 'full')
            allowed_users: List of user IDs to include (for train/val/test splits)
            norm_stats: Tuple of (mean, std) for normalization
            min_samples_per_user: Minimum samples per user to include that user
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.crop_frames = crop_frames
        self.selected_channels = selected_channels
        self.mode = mode
        self.allowed_users = allowed_users
        self.norm_stats = norm_stats
        self.min_samples_per_user = min_samples_per_user

        # Load and process data
        self._load_data()
        self._filter_users()
        self._create_samples()

        print(f"Loaded {len(self.samples)} samples from {len(self.user_to_samples)} users in {mode} mode")

    def _load_data(self):
        """Load the raw EEG data from files."""
        # This will depend on your data format
        # Example assuming your data is stored as pickle files or similar

        data_file = self.data_path / f"{self.dataset_name}.pkl"  # Adjust extension as needed

        if not data_file.exists():
            # Try different extensions
            for ext in ['.pkl', '.npz', '.h5', '.mat']:
                data_file = self.data_path / f"{self.dataset_name}{ext}"
                if data_file.exists():
                    break
            else:
                raise FileNotFoundError(f"Data file not found: {self.data_path / self.dataset_name}")

        # Load data based on file type
        if data_file.suffix == '.pkl':
            with open(data_file, 'rb') as f:
                self.raw_data = pickle.load(f)
        elif data_file.suffix == '.npz':
            self.raw_data = np.load(data_file, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        # Expected data format:
        # self.raw_data should be a dict with structure like:
        # {
        #     'data': np.array of shape (n_samples, n_channels, n_timepoints),
        #     'labels': np.array of shape (n_samples,) with user IDs,
        #     'channel_names': List[str] of channel names
        # }
        # OR
        # {
        #     'user_0': {'data': np.array, 'channel_names': List[str]},
        #     'user_1': {'data': np.array, 'channel_names': List[str]},
        #     ...
        # }

        # Adapt this based on your actual data format
        if 'data' in self.raw_data and 'labels' in self.raw_data:
            # Format 1: All data in single arrays
            self.eeg_data = self.raw_data['data']
            self.user_labels = self.raw_data['labels']
            self.channel_names = self.raw_data.get('channel_names', [f'Ch_{i}' for i in range(self.eeg_data.shape[1])])
        else:
            # Format 2: Data organized by user
            self._load_user_organized_data()

    def _load_user_organized_data(self):
        """Load data that's organized by user in separate files/keys."""
        all_data = []
        all_labels = []

        user_keys = [k for k in self.raw_data.keys() if k.startswith('user_') or k.isdigit()]

        for user_key in sorted(user_keys):
            if isinstance(user_key, str) and user_key.startswith('user_'):
                user_id = int(user_key.split('_')[1])
            else:
                user_id = int(user_key)

            user_data = self.raw_data[user_key]
            if isinstance(user_data, dict):
                eeg = user_data['data']  # Shape: (n_samples, n_channels, n_timepoints)
            else:
                eeg = user_data  # Direct array

            # Add user labels
            user_labels = np.full(len(eeg), user_id)

            all_data.append(eeg)
            all_labels.append(user_labels)

        self.eeg_data = np.concatenate(all_data, axis=0)
        self.user_labels = np.concatenate(all_labels, axis=0)

        # Get channel names from first user or create default
        first_user_data = list(self.raw_data.values())[0]
        if isinstance(first_user_data, dict) and 'channel_names' in first_user_data:
            self.channel_names = first_user_data['channel_names']
        else:
            self.channel_names = [f'Ch_{i}' for i in range(self.eeg_data.shape[1])]

    def _filter_users(self):
        """Filter users based on minimum samples and allowed users."""
        # Count samples per user
        unique_users, counts = np.unique(self.user_labels, return_counts=True)

        # Filter by minimum samples
        valid_users = unique_users[counts >= self.min_samples_per_user]

        # Filter by allowed users if specified
        if self.allowed_users is not None:
            valid_users = np.intersect1d(valid_users, self.allowed_users)

        # Create mask for valid samples
        valid_mask = np.isin(self.user_labels, valid_users)

        # Filter data
        self.eeg_data = self.eeg_data[valid_mask]
        self.user_labels = self.user_labels[valid_mask]

        # Remap user labels to be contiguous starting from 0
        self.unique_users = np.sort(np.unique(self.user_labels))
        self.user_mapping = {old_id: new_id for new_id, old_id in enumerate(self.unique_users)}
        self.user_labels = np.array([self.user_mapping[user_id] for user_id in self.user_labels])

        self.num_users = len(self.unique_users)

        print(f"Filtered to {self.num_users} users with at least {self.min_samples_per_user} samples each")

    def _create_samples(self):
        """Create the final list of samples with user grouping."""
        # Group samples by user for easier access
        self.user_to_samples = {}
        for idx, user_id in enumerate(self.user_labels):
            if user_id not in self.user_to_samples:
                self.user_to_samples[user_id] = []
            self.user_to_samples[user_id].append(idx)

        # Create flat list of all valid samples
        self.samples = list(range(len(self.eeg_data)))

        # Compute normalization stats if not provided
        if self.norm_stats is None and self.mode in ['train', 'full']:
            # Calculate stats on selected channels only
            selected_data = self.eeg_data[:, self.selected_channels, :]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            self.norm_stats = (mean, std)
            print(f"Computed normalization stats: mean={mean:.4f}, std={std:.4f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            eeg_data: Tensor of shape (8, crop_frames) - 8 selected EEG channels
            user_id: Tensor with user ID (scalar)
        """
        sample_idx = self.samples[idx]

        # Get EEG data for selected channels
        eeg_data = self.eeg_data[sample_idx, self.selected_channels, :]  # Shape: (8, n_timepoints)
        user_id = self.user_labels[sample_idx]

        # Crop/sample frames
        if eeg_data.shape[1] > self.crop_frames:
            # Random crop for training, center crop for val/test
            if self.mode == 'train':
                start_idx = np.random.randint(0, eeg_data.shape[1] - self.crop_frames + 1)
            else:
                start_idx = (eeg_data.shape[1] - self.crop_frames) // 2
            eeg_data = eeg_data[:, start_idx:start_idx + self.crop_frames]
        elif eeg_data.shape[1] < self.crop_frames:
            # Pad if too short
            pad_amount = self.crop_frames - eeg_data.shape[1]
            eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_amount)), mode='edge')

        # Apply normalization
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            eeg_data = (eeg_data - mean) / (std + 1e-8)

        # Convert to tensors
        eeg_data = torch.from_numpy(eeg_data).float()
        user_id = torch.tensor(user_id, dtype=torch.long)

        return eeg_data, user_id

    def get_user_samples(self, user_id: int) -> List[int]:
        """Get all sample indices for a specific user."""
        return self.user_to_samples.get(user_id, [])

    def get_channel_names(self, selected_only: bool = True) -> List[str]:
        """Get channel names."""
        if selected_only:
            return [self.channel_names[i] for i in self.selected_channels]
        return self.channel_names