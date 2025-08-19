import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional, Set
import re
from collections import defaultdict

import torch
import torch.nn.functional as F


class EEGClassificationDataset(torch.utils.data.Dataset):
    """
    EEG Classification dataset for user identification.

    Loads specific channels for each subject/frame combination and returns
    (8_channel_data, user_id) pairs for supervised learning.
    """

    # Mapping from channel indices to channel names
    # You'll need to verify this mapping based on your specific EEG setup
    # Standard 64-channel EEG layouts vary, so adjust as needed
    CHANNEL_MAPPING = {
        3: 'Fc1',
        12: 'C2',
        13: 'C4',
        18: 'Cpz',
        50: 'P1',
        60: 'Po8',
        61: 'O1',
        64: 'Iz',
    }

    # Alternative: if your files use different naming, update this mapping
    # You can also dynamically detect channel names from your actual files

    def __init__(self, folder, files, crop_frames, selected_channels: List[int],
                 norm_stats=None, tfms=None, random_crop=True,
                 mode='train', allowed_users: Optional[List[int]] = None,
                 min_samples_per_user: int = 10):
        """
        Args:
            folder: Root folder containing the data
            files: List of file paths from CSV
            crop_frames: Number of time frames to crop
            selected_channels: List of 8 channel indices to use
            norm_stats: Normalization statistics [mean, std]
            tfms: Transform functions
            random_crop: Whether to randomly crop or use center crop
            mode: 'train', 'val', or 'test'
            allowed_users: List of allowed user IDs for this split
            min_samples_per_user: Minimum samples per user to include
        """
        super().__init__()

        self.folder = Path(folder)
        self.crop_frames = crop_frames
        self.selected_channels = selected_channels
        self.tfms = tfms
        self.random_crop = random_crop if mode == 'train' else False
        self.mode = mode
        self.norm_stats = norm_stats

        # Map channel indices to names
        self.selected_channel_names = [self.CHANNEL_MAPPING.get(ch, f'Ch_{ch}')
                                       for ch in selected_channels]

        print(f"Selected channels: {dict(zip(selected_channels, self.selected_channel_names))}")

        # Group files by subject and frame
        self.samples = self._group_files_by_subject_frame(files)

        # Filter users if needed
        if allowed_users is not None or min_samples_per_user > 0:
            self.samples = self._filter_users(self.samples, allowed_users, min_samples_per_user)

        # Create final sample list
        self.sample_list = list(self.samples.keys())

        # Get user info
        self.users = sorted(set(sample[0] for sample in self.sample_list))
        self.num_users = len(self.users)
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}

        print(f'Dataset contains {len(self.sample_list)} samples from {self.num_users} users')
        print(f'Users: {self.users[:10]}{"..." if len(self.users) > 10 else ""}')

    def _group_files_by_subject_frame(self, files: List[str]) -> Dict[Tuple[int, str, int], Dict[str, str]]:
        """
        Group files by (subject_id, session, frame_id) and channel.

        Returns:
            Dict mapping (subject_id, session, frame_id) to {channel_name: file_path}
        """
        samples = defaultdict(dict)

        pattern = r'S(\d+)/S\d+R(\d+)/([A-Za-z0-9]+)_frame_(\d+)\.npy'

        for file_path in tqdm(files, desc="Grouping files by subject/frame"):
            match = re.match(pattern, file_path)
            if match:
                subject_id = int(match.group(1))
                session = f"R{match.group(2)}"
                channel_name = match.group(3)
                frame_id = int(match.group(4))

                # Only include files for our selected channels
                if channel_name in self.selected_channel_names:
                    key = (subject_id, session, frame_id)
                    samples[key][channel_name] = file_path
            else:
                print(f"Warning: Could not parse filename: {file_path}")

        # Filter out incomplete samples (must have all selected channels)
        complete_samples = {}
        for key, channels in samples.items():
            if len(channels) == len(self.selected_channel_names):
                # Check that all required channels are present
                if all(ch_name in channels for ch_name in self.selected_channel_names):
                    complete_samples[key] = channels

        print(f"Found {len(complete_samples)} complete samples with all {len(self.selected_channel_names)} channels")
        return complete_samples

    def _filter_users(self, samples: Dict, allowed_users: Optional[List[int]],
                      min_samples_per_user: int) -> Dict:
        """Filter samples based on user constraints."""
        # Count samples per user
        user_counts = defaultdict(int)
        for (subject_id, session, frame_id), channels in samples.items():
            user_counts[subject_id] += 1

        # Filter by minimum samples
        valid_users = {user for user, count in user_counts.items()
                       if count >= min_samples_per_user}

        # Filter by allowed users
        if allowed_users is not None:
            valid_users = valid_users.intersection(set(allowed_users))

        # Filter samples
        filtered_samples = {
            key: channels for key, channels in samples.items()
            if key[0] in valid_users  # key[0] is subject_id
        }

        print(f"Filtered to {len(valid_users)} users with at least {min_samples_per_user} samples")
        print(f"Kept {len(filtered_samples)} samples after filtering")

        return filtered_samples

    def __len__(self):
        return len(self.sample_list)

    def get_eeg_data(self, sample_key: Tuple[int, str, int]) -> torch.Tensor:
        """
        Load EEG data for all selected channels for a given sample.

        Returns:
            Tensor of shape (n_channels, n_timepoints)
        """
        subject_id, session, frame_id = sample_key
        channel_files = self.samples[sample_key]

        channel_data = []

        # Load data for each selected channel in order
        for channel_name in self.selected_channel_names:
            file_path = self.folder / channel_files[channel_name]
            try:
                data = torch.from_numpy(np.load(file_path))  # Shape: (n_timepoints,) or (1, n_timepoints)

                # Ensure 1D
                if data.dim() > 1:
                    data = data.squeeze()

                channel_data.append(data)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                # Use zeros as fallback
                channel_data.append(torch.zeros(self.crop_frames))

        # Stack channels
        eeg_data = torch.stack(channel_data, dim=0)  # Shape: (8, n_timepoints)
        return eeg_data

    def complete_eeg(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Process EEG data: crop/pad, normalize, apply transforms.
        Similar to complete_audio in original dataset.
        """
        # Current shape: (n_channels, n_timepoints)
        n_timepoints = eeg_data.shape[-1]

        # Crop or pad to desired length
        if n_timepoints > self.crop_frames:
            if self.random_crop:
                start = int(torch.randint(n_timepoints - self.crop_frames, (1,))[0])
            else:
                start = (n_timepoints - self.crop_frames) // 2
            eeg_data = eeg_data[..., start:start + self.crop_frames]
        elif n_timepoints < self.crop_frames:
            pad_amount = self.crop_frames - n_timepoints
            eeg_data = F.pad(eeg_data, (0, pad_amount), mode='constant', value=0)

        # Convert to float
        eeg_data = eeg_data.to(torch.float)

        # Normalize
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            eeg_data = (eeg_data - mean) / std

        # Apply transforms
        if self.tfms is not None:
            eeg_data = self.tfms(eeg_data)

        return eeg_data

    def __getitem__(self, index):
        """
        Returns:
            eeg_data: Tensor of shape (8, crop_frames) - 8 selected EEG channels
            user_id: Tensor with user ID (scalar)
        """
        sample_key = self.sample_list[index]
        subject_id, session, frame_id = sample_key

        # Load EEG data for all selected channels
        eeg_data = self.get_eeg_data(sample_key)

        # Process the data (crop, normalize, etc.)
        eeg_data = self.complete_eeg(eeg_data)

        # Convert subject_id to user index (0-based)
        user_id = torch.tensor(self.user_to_idx[subject_id], dtype=torch.long)

        return eeg_data, user_id

    def get_sample_info(self, index: int) -> Dict:
        """Get information about a specific sample."""
        sample_key = self.sample_list[index]
        subject_id, session, frame_id = sample_key

        return {
            'subject_id': subject_id,
            'session': session,
            'frame_id': frame_id,
            'user_index': self.user_to_idx[subject_id],
            'files': self.samples[sample_key]
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'samples={len(self.sample_list)}, '
                f'users={self.num_users}, '
                f'channels={self.selected_channels}, '
                f'crop_frames={self.crop_frames}, '
                f'random_crop={self.random_crop})')


def get_files(dataset_name):
    """Load file list from CSV, compatible with original function."""
    files = pd.read_csv(str(dataset_name)).file_name.values
    files = sorted(files)
    return files


def build_classification_dataset(data_path: str,
                                 dataset: str,
                                 crop_frames: int,
                                 selected_channels: List[int],
                                 norm_stats: Tuple[float, float] = None,
                                 mode: str = 'train',
                                 allowed_users: Optional[List[int]] = None,
                                 min_samples_per_user: int = 10):
    """
    Build EEG classification dataset.

    Args:
        data_path: Root folder of the dataset
        dataset: Path to CSV file with file list
        crop_frames: Number of time frames to crop
        selected_channels: List of 8 channel indices to use
        norm_stats: Normalization statistics [mean, std]
        mode: 'train', 'val', or 'test'
        allowed_users: List of allowed user IDs for this split
        min_samples_per_user: Minimum samples per user
    """
    files = get_files(dataset)

    ds = EEGClassificationDataset(
        folder=data_path,
        files=files,
        crop_frames=crop_frames,
        selected_channels=selected_channels,
        norm_stats=norm_stats,
        tfms=None,  # No transforms for now
        random_crop=(mode == 'train'),
        mode=mode,
        allowed_users=allowed_users,
        min_samples_per_user=min_samples_per_user
    )

    return ds