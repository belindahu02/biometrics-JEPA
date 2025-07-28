"""
EEG to log-mel spectrogram (LMS) converter with 20s frames and 10s overlap.
This program converts EEG .edf files found in the source folder to log-mel spectrograms,
then creates 20-second frames with 10-second overlap from each channel's spectrogram.
Each frame is saved as a separate .npy file.

The conversion includes the following processes:
    - Multi-channel EEG processing (saves each channel separately)
    - Resampling to target sampling rate
    - Converting to a log-mel spectrogram with same dimensions as audio converter
    - Segmenting into 20s frames with 10s overlap

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

    # Frame parameters
    frame_duration = 20.0  # 20 seconds per frame
    frame_overlap = 10.0   # 10 seconds overlap
    frame_stride = frame_duration - frame_overlap  # 10 seconds stride


def create_overlapping_frames(spectrogram, frame_duration, frame_stride, hop_size, sample_rate):
    """
    Create overlapping frames from a spectrogram.

    Args:
        spectrogram: Input spectrogram tensor of shape (1, n_mels, time_frames)
        frame_duration: Duration of each frame in seconds
        frame_stride: Stride between frames in seconds
        hop_size: Hop size used in spectrogram computation
        sample_rate: Original sampling rate

    Returns:
        List of spectrogram frames, each of shape (1, n_mels, frame_time_frames)
    """
    batch_size, n_mels, total_time_frames = spectrogram.shape

    # Calculate frame dimensions in spectrogram time frames
    frames_per_second = sample_rate / hop_size
    frame_length_frames = int(frame_duration * frames_per_second)
    frame_stride_frames = int(frame_stride * frames_per_second)

    frames = []
    start_frame = 0

    while start_frame + frame_length_frames <= total_time_frames:
        end_frame = start_frame + frame_length_frames
        frame = spectrogram[:, :, start_frame:end_frame]
        frames.append(frame)
        start_frame += frame_stride_frames

    return frames


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
        total_frames_saved = 0

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

                # Calculate minimum length for meaningful frames
                min_samples_for_frames = int(prms.sample_rate * prms.frame_duration)

                # Pad if too short for at least one frame
                if len(eeg_signal) < min_samples_for_frames:
                    if verbose:
                        print(f'Padding {subpathname} channel {channel_idx} from {len(eeg_signal)} to {min_samples_for_frames} samples for frame processing')
                    eeg_signal = np.pad(eeg_signal, (0, min_samples_for_frames - len(eeg_signal)))

                # Convert to log-mel spectrogram
                lms = to_lms(eeg_signal)

                # Create overlapping frames
                frames = create_overlapping_frames(
                    lms,
                    prms.frame_duration,
                    prms.frame_stride,
                    prms.hop_size,
                    prms.sample_rate
                )

                # Save each frame as a separate file
                channel_frames_saved = 0
                for frame_idx, frame in enumerate(frames):
                    frame_filename = folder_name / f"{channel_name}_frame_{frame_idx:03d}.npy"
                    np.save(frame_filename, frame.numpy())
                    channel_frames_saved += 1
                    total_frames_saved += 1

                if len(frames) > 0:
                    successful_channels.append(channel_name)
                    if verbose:
                        print(f'  Channel {channel_idx} ({channel_name}) -> {channel_frames_saved} frames, each shape: {frames[0].shape}')
                else:
                    if verbose:
                        print(f'  Channel {channel_idx} ({channel_name}) -> No frames generated (signal too short)')

            except Exception as e:
                print(f'ERROR processing channel {channel_idx} in {subpathname}: {str(e)}')
                continue

        if verbose:
            print(f'Saved {total_frames_saved} total frames from {len(successful_channels)}/{n_channels} channels for {edf_input_path.name}')

    except Exception as e:
        print('ERROR failed to open or convert', edf_input_path.name, '-', str(e))
        return f'{edf_input_path.name} (failed)'

    return f'{file_stem} ({len(successful_channels)}/{n_channels} channels, {total_frames_saved} frames)'


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
                      suffix='.edf', skip=0, min_length=20.0, verbose=False) -> None:
    """
    Convert EEG files from a range of subjects to log-mel spectrograms with 20s frames and 10s overlap.

    Args:
        base_from_dir: Base source directory (e.g., /workspace/input_eeg_data/physionet.org/files/eegmmidb/1.0.0/)
        base_to_dir: Base destination directory for .npy files (e.g., /workspace/output_spectrograms)
        start_subject_id: Starting subject ID (inclusive, e.g., 1 for S001)
        end_subject_id: Ending subject ID (inclusive, e.g., 109 for S109)
        suffix: File extension to process (default: '.edf')
        skip: Number of files to skip per subject (default: 0)
        min_length: Minimum length in seconds for processing (default: 20.0 for at least one frame)
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
    print(f'Processing ALL channels separately with 20s frames and 10s overlap')
    print(f'Frame duration: {prms.frame_duration}s, Frame stride: {prms.frame_stride}s')
    print(f'Output shape per frame: (80, ~{int(prms.frame_duration * prms.sample_rate / prms.hop_size)})')
    print(f'Frequency range: {prms.f_min}-{prms.f_max} Hz')

    for i in range(start_subject_id, end_subject_id + 1):
        subject_id = f"S{i:03d}"  # Format as S001, S002, etc.

        # Construct the full source and destination paths for the current subject
        # Input: base_from_dir/S001/...edf
        # Output: base_to_dir/S001/...npy
        current_subject_from_dir = base_from_path / subject_id
        current_subject_to_dir = base_to_path / subject_id

        if not current_subject_from_dir.exists():
            print(f"WARNING: Source directory for {subject_id} not found: {current_subject_from_dir}. Skipping.")
            continue

        print(f"\n--- Processing Subject: {subject_id} ---")

        # Find all EDF files for the current subject
        subject_edf_files = [f.relative_to(current_subject_from_dir) for f in current_subject_from_dir.glob(f'**/*{suffix}')]
        subject_edf_files = sorted(subject_edf_files)

        if skip > 0:
            subject_edf_files = subject_edf_files[skip:]

        if len(subject_edf_files) == 0:
            print(f'No {suffix} files found for {subject_id} in {current_subject_from_dir}')
            continue

        # Prepare arguments for multiprocessing worker for this subject's files
        worker_args_for_subject = []
        for f in subject_edf_files:
            worker_args_for_subject.append([f, str(current_subject_from_dir), str(current_subject_to_dir), prms, to_lms, suffix, min_length, verbose])

        print(f'  Processing {len(subject_edf_files)} {suffix} files for {subject_id}...')

        # Process files for this subject
        with Pool() as p:
            results = list(tqdm(p.imap(_converter_worker, worker_args_for_subject), total=len(worker_args_for_subject)))

        # Collect paths for CSV for the current subject
        for subpathname_relative_to_subject_dir, from_dir_worker, to_dir_worker, _, _, _, _, _ in worker_args_for_subject:
            file_stem = Path(subpathname_relative_to_subject_dir).stem
            output_subfolder_name = Path(subpathname_relative_to_subject_dir).parent / file_stem
            full_output_folder_path_in_container = Path(to_dir_worker) / output_subfolder_name

            if full_output_folder_path_in_container.exists():
                for npy_file in full_output_folder_path_in_container.glob('*.npy'):
                    relative_path_from_base_output = npy_file.relative_to(base_to_path)
                    all_generated_file_paths.append(relative_path_from_base_output)

        successful_subject_files = [r for r in results if r and 'failed' not in r]
        print(f'  Finished {subject_id}. Successfully processed {len(successful_subject_files)}/{len(subject_edf_files)} files.')
        for result in results:
            if verbose and result:
                print(f'    {result}')

    # Write to CSV after all subjects are processed
    all_generated_file_paths.sort()

    csv_filename = base_to_path / "files_audioset.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['file_name'])
            for filepath in all_generated_file_paths:
                csv_writer.writerow([str(filepath)])
        print(f"\nSuccessfully wrote ALL generated file paths to {csv_filename}")
        print(f"Total files generated: {len(all_generated_file_paths)}")
    except Exception as e:
        print(f"ERROR writing to CSV file {csv_filename}: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    fire.Fire(convert_eeg_batch)
    convert_eeg_batch("/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0",
                      "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data", 1, 2)
