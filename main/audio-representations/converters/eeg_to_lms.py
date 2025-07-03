"""
EEG to log-mel spectrogram (LMS) converter.
This program converts EEG .edf files found in the source folder to log-mel spectrograms,
then stores them in the destination folder while holding the same relative path structure.
The conversion includes the following processes:
    - Multi-channel EEG processing (saves each channel separately)
    - Resampling to target sampling rate
    - Converting to a log-mel spectrogram with same dimensions as audio converter
Example:
    python eeg_to_lms.py /your/eeg/data /your/eeg_lms_output
"""
import numpy as np
from pathlib import Path
import mne
from multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import fire
from tqdm import tqdm
import nnAudio.features
import warnings
from scipy import signal
import os
import csv

warnings.simplefilter('ignore')


class EEG_FFT_parameters:
    # Match the audio converter parameters for consistent output dimensions
    sample_rate = 16000  # Target sampling rate (will resample EEG to this)
    window_size = 400  # 25ms window at 16kHz
    n_fft = 400
    hop_size = 160  # 10ms stride at 16kHz
    n_mels = 80  # Number of mel frequency bins
    f_min = 0.5  # Lower frequency limit for EEG (Hz)
    f_max = 100  # Upper frequency limit for EEG (Hz) - typical EEG range


def _converter_worker(args):
    subpathname, from_dir, to_dir, prms, to_lms, suffix, min_length, verbose = args
    from_dir_path_obj, to_dir_path_obj = Path(from_dir), Path(to_dir) # Use Path objects inside worker

    # Reconstruct the full path to the EDF file inside the container's mounted input dir
    # subpathname is relative to the *original* from_dir (e.g., "S072/S072R10.edf")
    edf_input_path = from_dir_path_obj / subpathname

    # Create folder name from file name (without extension)
    file_stem = Path(subpathname).stem  # e.g., "S001R01" from "S001R01.edf"
    # Ensure the output folder structure mirrors the input structure under the mounted to_dir
    # Example: if subpathname is "S072/S072R10.edf", folder_name will be "output_mount_point/S072/S072R10"
    relative_output_path = Path(subpathname).parent / file_stem
    folder_name = to_dir_path_obj / relative_output_path


    # Check if folder already exists and has files
    if folder_name.exists() and any(folder_name.glob('*.npy')):
        if verbose:
            print(f'Folder {file_stem} already exists with files at {folder_name}')
        return f'{file_stem} (already exists)'

    # Load and convert EEG to log-mel spectrogram
    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(str(edf_input_path), preload=True, verbose=False)

        # Get EEG data and channel names
        eeg_data = raw.get_data()  # Shape: (n_channels, n_samples)
        channel_names = raw.ch_names
        orig_sfreq = raw.info['sfreq']
        n_channels = eeg_data.shape[0]

        if verbose:
            print(f'Processing {edf_input_path.name}: {n_channels} channels, {len(channel_names)} channel names')

        # Create output folder
        folder_name.mkdir(parents=True, exist_ok=True)

        # Process each channel separately
        successful_channels = []
        lms_shape = None # Initialize to store shape
        for channel_idx in range(n_channels):
            try:
                # Get channel data
                eeg_signal = eeg_data[channel_idx, :]

                # Get channel name (use index if name not available)
                if channel_idx < len(channel_names):
                    channel_name = channel_names[channel_idx]
                    # Clean channel name for filename (remove invalid characters)
                    channel_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    channel_name = channel_name.replace(' ', '_')
                else:
                    channel_name = f"channel_{channel_idx:02d}"

                # Resample to target sampling rate if needed
                if orig_sfreq != prms.sample_rate:
                    # Calculate resampling ratio
                    resample_ratio = prms.sample_rate / orig_sfreq
                    n_samples_new = int(len(eeg_signal) * resample_ratio)
                    eeg_signal = signal.resample(eeg_signal, n_samples_new)

                # Normalize EEG signal (important for spectrogram quality)
                eeg_signal = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)

                # Pad if too short
                if min_length is not None:
                    min_samples = int(prms.sample_rate * min_length)
                    if len(eeg_signal) < min_samples:
                        if verbose:
                            print(f'Padding {subpathname} channel {channel_idx} from {len(eeg_signal)} to {min_samples} samples')
                        eeg_signal = np.pad(eeg_signal, (0, min_samples - len(eeg_signal)))

                # Convert to log-mel spectrogram
                lms = to_lms(eeg_signal)
                lms_shape = lms.shape # Update shape

                # Save the spectrogram for this channel
                channel_filename = folder_name / f"{channel_name}.npy"
                np.save(channel_filename, lms.numpy())
                successful_channels.append(channel_name)

                if verbose:
                    print(f'  Channel {channel_idx} ({channel_name}) -> {channel_filename}, shape: {lms.shape}')

            except Exception as e:
                print(f'ERROR processing channel {channel_idx} in {subpathname}: {str(e)}')
                continue

        if verbose:
            print(f'Saved {len(successful_channels)}/{n_channels} channels for {edf_input_path.name}')

    except Exception as e:
        print('ERROR failed to open or convert', edf_input_path.name, '-', str(e))
        return f'{edf_input_path.name} (failed)'

    return f'{file_stem} ({len(successful_channels)}/{n_channels} channels)'


class ToLogMelSpec:
    def __init__(self, cfg):
        self.cfg = cfg
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.window_size,
            hop_length=cfg.hop_size,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def __call__(self, signal):
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)

        if signal.dim() > 1:
            signal = signal.squeeze()

        x = self.to_spec(signal)
        x = (x + torch.finfo(torch.float32).eps).log()

        return x


def convert_eeg_batch(base_from_dir, base_to_dir,
                      start_subject_id=1, end_subject_id=109,
                      suffix='.edf', skip=0, min_length=6.1, verbose=False) -> None:
    """
    Convert EEG files from a range of subjects to log-mel spectrograms.

    Args:
        base_from_dir: Base source directory (e.g., /workspace/input_eeg_data/physionet.org/files/eegmmidb/1.0.0/)
        base_to_dir: Base destination directory for .npy files (e.g., /workspace/output_spectrograms)
        start_subject_id: Starting subject ID (inclusive, e.g., 1 for S001)
        end_subject_id: Ending subject ID (inclusive, e.g., 109 for S109)
        suffix: File extension to process (default: '.edf')
        skip: Number of files to skip per subject (default: 0)
        min_length: Minimum length in seconds (default: 6.1)
        verbose: Print detailed progress information
    """
    base_from_path = Path(base_from_dir)
    base_to_path = Path(base_to_dir)
    all_generated_file_paths = []

    prms = EEG_FFT_parameters()
    to_lms = ToLogMelSpec(prms)

    print(f'Starting batch conversion for subjects S{start_subject_id:03d} to S{end_subject_id:03d}')
    print(f'Base Input Directory: {base_from_dir}')
    print(f'Base Output Directory: {base_to_dir}')
    print(f'Target sampling rate: {prms.sample_rate} Hz')
    print(f'Processing ALL channels separately')
    print(f'Output shape per channel: (80, time_frames)')
    print(f'Frequency range: {prms.f_min}-{prms.f_max} Hz')


    for i in range(start_subject_id, end_subject_id + 1):
        subject_id = f"S{i:03d}"  # Format as S001, S002, etc.

        # Construct the full source and destination paths for the current subject
        # Input: base_from_dir/S001/...edf
        # Output: base_to_dir/S001/...npy
        current_subject_from_dir = base_from_path / subject_id
        current_subject_to_dir = base_to_path / subject_id # This creates output as base_to_dir/S001/S001RXX/channel.npy

        if not current_subject_from_dir.exists():
            print(f"WARNING: Source directory for {subject_id} not found: {current_subject_from_dir}. Skipping.")
            continue

        print(f"\n--- Processing Subject: {subject_id} ---")

        # Find all EDF files for the current subject, relative to current_subject_from_dir
        # This will yield paths like "S001R01.edf", "S001R02.edf", etc.
        subject_edf_files = [f.relative_to(current_subject_from_dir) for f in current_subject_from_dir.glob(f'**/*{suffix}')]
        subject_edf_files = sorted(subject_edf_files)

        if skip > 0:
            subject_edf_files = subject_edf_files[skip:]

        if len(subject_edf_files) == 0:
            print(f'No {suffix} files found for {subject_id} in {current_subject_from_dir}')
            continue

        # Prepare arguments for multiprocessing worker for this subject's files
        # The worker needs to know the *base* mounted directories (from_dir, to_dir) to construct absolute paths
        # correctly for the *current file*.
        worker_args_for_subject = []
        for f in subject_edf_files:
            # The 'subpathname' for the worker needs to be relative to the *root of the dataset*
            # For example, if base_from_dir is /workspace/input_eeg_data/physionet.org/files/eegmmidb/1.0.0/
            # and current_subject_from_dir is .../S001/
            # and f is S001R01.edf
            # we need subpathname to be S001/S001R01.edf for the worker to find it relative to base_from_dir.
            # However, the current worker logic assumes 'subpathname' is relative to 'from_dir'.
            # Let's adjust the worker call slightly or how 'subpathname' is generated.

            # Option 1: Pass the full path to the worker, and let the worker strip it. Simpler.
            # No, the worker needs subpathname to calculate relative output paths.

            # Option 2 (Better): Keep subpathname relative to the subject's directory.
            # And pass the subject's specific from_dir and to_dir to the worker.
            # This is how your original _converter_worker was designed.
            worker_args_for_subject.append([f, str(current_subject_from_dir), str(current_subject_to_dir), prms, to_lms, suffix, min_length, verbose])


        print(f'  Processing {len(subject_edf_files)} {suffix} files for {subject_id}...')

        # Process files for this subject
        with Pool() as p:
            results = list(tqdm(p.imap(_converter_worker, worker_args_for_subject), total=len(worker_args_for_subject)))

        # Collect paths for CSV for the current subject
        for subpathname_relative_to_subject_dir, from_dir_worker, to_dir_worker, _, _, _, _, _ in worker_args_for_subject:
            file_stem = Path(subpathname_relative_to_subject_dir).stem
            # This is the path like "S001R01" which is the folder name
            output_subfolder_name = Path(subpathname_relative_to_subject_dir).parent / file_stem

            # The full path to the output folder for this file within the container
            full_output_folder_path_in_container = Path(to_dir_worker) / output_subfolder_name

            if full_output_folder_path_in_container.exists():
                for npy_file in full_output_folder_path_in_container.glob('*.npy'):
                    # The path added to the CSV should be relative to the *base_to_dir*
                    # Example: S001/S001R01/EEG_Channel_Name.npy
                    relative_path_from_base_output = npy_file.relative_to(base_to_path)
                    all_generated_file_paths.append(relative_path_from_base_output)

        successful_subject_files = [r for r in results if r and 'failed' not in r]
        print(f'  Finished {subject_id}. Successfully processed {len(successful_subject_files)}/{len(subject_edf_files)} files.')
        for result in results:
            if verbose and result: # Only print detailed results if verbose is true
                print(f'    {result}')


    # Write to CSV after all subjects are processed
    all_generated_file_paths.sort() # Sort all paths before writing

    csv_filename = base_to_path / "files_audioset.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filepath']) # Header row
            for filepath in all_generated_file_paths:
                csv_writer.writerow([str(filepath)]) # Write each path as a row
        print(f"\nSuccessfully wrote ALL generated file paths to {csv_filename}")
    except Exception as e:
        print(f"ERROR writing to CSV file {csv_filename}: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # The default command will run the batch conversion
    # Pass base_from_dir and base_to_dir from Docker mounts
    fire.Fire(convert_eeg_batch)