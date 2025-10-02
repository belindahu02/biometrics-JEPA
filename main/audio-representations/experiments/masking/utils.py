"""
Masking utilities for EEG signal processing
Handles conversion from raw EDF files to masked spectrograms
"""

import numpy as np
import mne
from scipy import signal
import torch
import nnAudio.features
from pathlib import Path
import warnings
import os

warnings.simplefilter('ignore')


class EEGMaskingProcessor:
    """Process EEG files with masking applied before spectrogram conversion"""

    def __init__(self, sample_rate=16000, frame_duration=20.0, frame_stride=10.0):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_stride = frame_stride

        # Spectrogram parameters (from your EEG_FFT_parameters)
        self.window_size = 400  # 25ms window at 16kHz
        self.n_fft = 400
        self.hop_size = 160  # 10ms stride at 16kHz
        self.n_mels = 80  # Number of mel frequency bins
        self.f_min = 0.5
        self.f_max = 100

        # Initialize mel spectrogram converter
        self.to_lms = nnAudio.features.MelSpectrogram(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            center=True,
            power=2,
            verbose=False,
        )

        print(f"EEG Masking Processor initialized:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Frame duration: {self.frame_duration}s")
        print(f"  Frame stride: {self.frame_stride}s")
        print(f"  Mel bins: {self.n_mels}")

    def apply_masking(self, signal_frame, masking_percentage, num_blocks):
        """
        Apply masking to a signal frame

        Args:
            signal_frame: 1D numpy array of signal
            masking_percentage: Total percentage of frame to mask (0-100)
            num_blocks: Number of separate blocks to mask

        Returns:
            masked_frame: Signal with masking applied
        """
        if masking_percentage == 0 or num_blocks == 0:
            return signal_frame.copy()

        frame_masked = signal_frame.copy()
        frame_length = len(signal_frame)

        # Calculate total samples to mask
        total_mask_samples = int((masking_percentage / 100.0) * frame_length)

        if total_mask_samples <= 0:
            return frame_masked

        # Calculate samples per block
        samples_per_block = total_mask_samples // num_blocks
        remaining_samples = total_mask_samples % num_blocks

        if samples_per_block == 0:
            # If blocks are too small, just mask random individual samples
            mask_indices = np.random.choice(frame_length, size=total_mask_samples, replace=False)
            frame_masked[mask_indices] = 0.0
            return frame_masked

        # Apply blocks of masking
        for i in range(num_blocks):
            # Current block size (distribute remaining samples to first blocks)
            current_block_size = samples_per_block + (1 if i < remaining_samples else 0)

            if current_block_size >= frame_length:
                # Block too large, mask entire frame
                frame_masked[:] = 0.0
                break

            # Find random position for this block
            max_start = frame_length - current_block_size
            if max_start <= 0:
                continue

            start_pos = np.random.randint(0, max_start + 1)
            end_pos = start_pos + current_block_size

            # Apply masking (set to zero)
            frame_masked[start_pos:end_pos] = 0.0

        return frame_masked

    def split_into_frames(self, signal):
        """Split 1D signal into overlapping frames"""
        frame_length = int(self.frame_duration * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        frames = []

        start = 0
        while start + frame_length <= len(signal):
            frames.append(signal[start:start + frame_length])
            start += frame_step

        return frames

    def signal_to_spectrogram(self, signal_frame):
        """Convert signal frame to log-mel spectrogram"""
        if isinstance(signal_frame, np.ndarray):
            signal_frame = torch.tensor(signal_frame, dtype=torch.float32)

        if signal_frame.dim() > 1:
            signal_frame = signal_frame.squeeze()

        # Convert to mel spectrogram
        spec = self.to_lms(signal_frame)

        # Convert to log scale
        log_spec = (spec + torch.finfo(torch.float32).eps).log()

        # Crop to expected time frames (2000)
        if log_spec.shape[-1] > 2000:
            log_spec = log_spec[:, :, :2000]

        return log_spec

    def process_edf_file(self, edf_path, masking_percentage=0, num_blocks=1, output_dir=None):
        """
        Process a single EDF file with masking and convert to spectrograms

        Args:
            edf_path: Path to EDF file
            masking_percentage: Percentage of each frame to mask (0-100)
            num_blocks: Number of masking blocks per frame
            output_dir: Directory to save spectrograms (if None, returns in memory)

        Returns:
            Dictionary of channel spectrograms or saves to disk
        """
        edf_path = Path(edf_path)

        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")

        print(f"Processing {edf_path.name} with {masking_percentage}% masking ({num_blocks} blocks)")

        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            eeg_data = raw.get_data()
            channel_names = raw.ch_names
            orig_sfreq = raw.info['sfreq']
            n_channels = eeg_data.shape[0]

            results = {}

            for channel_idx in range(n_channels):
                eeg_signal = eeg_data[channel_idx, :]

                # Resample if needed
                if orig_sfreq != self.sample_rate:
                    resample_ratio = self.sample_rate / orig_sfreq
                    n_samples_new = int(len(eeg_signal) * resample_ratio)
                    eeg_signal = signal.resample(eeg_signal, n_samples_new)

                # Normalize
                eeg_signal = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)

                # Pad if too short
                min_samples = int(self.sample_rate * self.frame_duration)
                if len(eeg_signal) < min_samples:
                    eeg_signal = np.pad(eeg_signal, (0, min_samples - len(eeg_signal)))

                # Split into frames
                frames = self.split_into_frames(eeg_signal)

                # Process each frame
                channel_name = channel_names[channel_idx] if channel_idx < len(
                    channel_names) else f"channel_{channel_idx:02d}"
                channel_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                channel_name = channel_name.replace(' ', '_')

                channel_spectrograms = []

                for frame_idx, frame in enumerate(frames):
                    # Apply masking
                    masked_frame = self.apply_masking(frame, masking_percentage, num_blocks)

                    # Convert to spectrogram
                    lms_frame = self.signal_to_spectrogram(masked_frame)

                    if output_dir is not None:
                        # Save to disk
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        frame_filename = output_dir / f"{channel_name}_frame_{frame_idx:03d}.npy"
                        np.save(frame_filename, lms_frame.numpy())
                    else:
                        # Store in memory
                        channel_spectrograms.append(lms_frame.numpy())

                if output_dir is None:
                    results[channel_name] = channel_spectrograms

            if output_dir is not None:
                print(f"Saved {n_channels} channels to {output_dir}")
                return str(output_dir)
            else:
                return results

        except Exception as e:
            print(f"Error processing {edf_path}: {e}")
            raise

    def process_session_data(self, raw_eeg_dir, user_id, session_nums, masking_percentage=0,
                             num_blocks=1, output_base_dir=None):
        """
        Process multiple sessions for a user with masking

        Args:
            raw_eeg_dir: Base directory containing raw EDF files
            user_id: User ID (1-109)
            session_nums: List of session numbers to process
            masking_percentage: Percentage to mask
            num_blocks: Number of masking blocks
            output_base_dir: Base directory for output

        Returns:
            Dictionary of results or output directory path
        """
        raw_eeg_dir = Path(raw_eeg_dir)
        user_folder = f"S{user_id:03d}"
        user_path = raw_eeg_dir / user_folder

        if not user_path.exists():
            raise FileNotFoundError(f"User directory not found: {user_path}")

        all_results = {}

        for session_num in session_nums:
            session_folder = f"{user_folder}R{session_num:02d}"

            # Find EDF files for this session
            edf_files = list(user_path.glob(f"{session_folder}*.edf"))

            if not edf_files:
                print(f"Warning: No EDF files found for {session_folder}")
                continue

            session_results = {}

            for edf_file in edf_files:
                if output_base_dir is not None:
                    session_output_dir = Path(output_base_dir) / user_folder / session_folder
                    result = self.process_edf_file(
                        edf_file, masking_percentage, num_blocks, session_output_dir
                    )
                else:
                    result = self.process_edf_file(
                        edf_file, masking_percentage, num_blocks, None
                    )

                session_results[edf_file.stem] = result

            all_results[session_folder] = session_results

        return all_results if output_base_dir is None else str(output_base_dir)


def create_masked_dataset_for_experiment(raw_eeg_dir, output_dir, user_ids, session_nums,
                                         masking_percentage, num_blocks):
    """
    Create a complete masked dataset for the experiment

    Args:
        raw_eeg_dir: Directory with raw EDF files
        output_dir: Where to save processed data
        user_ids: List of user IDs to process (1-10)
        session_nums: List of session numbers to process
        masking_percentage: Percentage to mask (0-50)
        num_blocks: Number of masking blocks (1-5)
    """
    processor = EEGMaskingProcessor()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating masked dataset:")
    print(f"  Users: S{min(user_ids):03d}-S{max(user_ids):03d} ({len(user_ids)} users)")
    print(f"  Sessions: {session_nums}")
    print(f"  Masking: {masking_percentage}% with {num_blocks} blocks")
    print(f"  Output: {output_dir}")

    successful_users = 0
    total_files_processed = 0

    for user_id in user_ids:
        try:
            print(f"Processing user S{user_id:03d}...")

            result = processor.process_session_data(
                raw_eeg_dir=raw_eeg_dir,
                user_id=user_id,
                session_nums=session_nums,
                masking_percentage=masking_percentage,
                num_blocks=num_blocks,
                output_base_dir=output_dir
            )

            successful_users += 1

            # Count files processed
            user_output_dir = output_dir / f"S{user_id:03d}"
            if user_output_dir.exists():
                files_count = len(list(user_output_dir.glob("**/*.npy")))
                total_files_processed += files_count
                print(f"  S{user_id:03d}: {files_count} files created")

        except Exception as e:
            print(f"  Error processing S{user_id:03d}: {e}")

    # Create CSV file listing all generated files
    csv_path = output_dir / "files_audioset.csv"
    create_file_listing_csv(output_dir, csv_path)

    print(f"Dataset creation completed:")
    print(f"  Successful users: {successful_users}/{len(user_ids)}")
    print(f"  Total files: {total_files_processed}")
    print(f"  CSV listing: {csv_path}")

    return str(output_dir)


def create_file_listing_csv(data_dir, csv_path):
    """Create CSV file listing all .npy files in the dataset"""
    data_dir = Path(data_dir)

    # Find all .npy files
    npy_files = []
    for npy_file in data_dir.glob("**/*.npy"):
        # Get relative path from data_dir
        rel_path = npy_file.relative_to(data_dir)
        npy_files.append(str(rel_path))

    # Write CSV
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name'])
        for file_path in sorted(npy_files):
            writer.writerow([file_path])

    print(f"Created CSV with {len(npy_files)} files: {csv_path}")


def run_embedding_extraction(csv_path, data_dir, embeddings_dir, model_checkpoint_path,
                             config_path=None, batch_size=16, device=None):
    """
    Extract embeddings using a pre-trained JEPA encoder model
    Uses your actual precompute.py functionality with pre-trained weights
    """
    from embedding_extractor import extract_embeddings_for_masking_experiment

    print(f"Extracting REAL embeddings from {data_dir} to {embeddings_dir}")
    print(f"Using checkpoint: {model_checkpoint_path}")

    if not model_checkpoint_path or not Path(model_checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint_path}")

    return extract_embeddings_for_masking_experiment(
        csv_file=csv_path,
        data_dir=data_dir,
        output_dir=embeddings_dir,
        checkpoint_path=model_checkpoint_path,
        config_path=config_path,
        batch_size=batch_size,
        device=device
    )


def run_embedding_grouping(embeddings_dir, grouped_dir):
    """
    Group embeddings by frame ID
    Adapted from your group.py
    """
    print(f"Grouping embeddings from {embeddings_dir} to {grouped_dir}")

    embeddings_dir = Path(embeddings_dir)
    grouped_dir = Path(grouped_dir)
    grouped_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict

    for root, dirs, files in os.walk(embeddings_dir):
        # Only process leaf folders with .npy files
        npy_files = [f for f in files if f.endswith("_emb.npy")]
        if not npy_files:
            continue

        # Prepare frame-wise grouping
        frames_dict = defaultdict(list)

        for f in npy_files:
            # Extract frame ID from filename: channel_frame_000_emb.npy -> 000
            parts = f.split("_frame_")
            if len(parts) >= 2:
                frame_id = parts[1].split("_")[0]
                frames_dict[frame_id].append(f)

        # Stack embeddings for each frame
        leaf_output_dir = grouped_dir / Path(root).relative_to(embeddings_dir)
        leaf_output_dir.mkdir(parents=True, exist_ok=True)

        for frame_id, frame_files in frames_dict.items():
            frame_files.sort()
            arrays = [np.load(Path(root) / f) for f in frame_files]
            stacked = np.vstack(arrays)
            save_path = leaf_output_dir / f"{frame_id}_stacked.npy"
            np.save(save_path, stacked)

        print(f"Processed {len(frames_dict)} frames from {root}")

    print(f"Embedding grouping completed")
    return str(grouped_dir)


def test_masking_processor():
    """Test the EEG masking processor"""
    print("Testing EEG Masking Processor...")

    # Create test signal
    duration = 20.0  # 20 seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))  # 10 Hz sine with noise

    processor = EEGMaskingProcessor()

    # Test different masking configurations
    test_configs = [
        (0, 1),  # No masking
        (10, 1),  # 10% masking, 1 block
        (10, 2),  # 10% masking, 2 blocks
        (25, 5),  # 25% masking, 5 blocks
    ]

    for mask_pct, num_blocks in test_configs:
        print(f"Testing {mask_pct}% masking with {num_blocks} blocks...")

        # Apply masking
        masked_signal = processor.apply_masking(test_signal, mask_pct, num_blocks)

        # Check masking percentage
        zero_samples = np.sum(masked_signal == 0)
        actual_pct = (zero_samples / len(masked_signal)) * 100
        print(f"  Expected: {mask_pct}%, Actual: {actual_pct:.1f}%")

        # Convert to spectrogram
        spec = processor.signal_to_spectrogram(masked_signal)
        print(f"  Spectrogram shape: {spec.shape}")

    print("✅ Masking processor test completed!")


if __name__ == "__main__":
    test_masking_processor()