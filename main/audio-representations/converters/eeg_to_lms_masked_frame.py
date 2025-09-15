"""
EEG to log-mel spectrogram (LMS) converter with 20s frames and 10s overlap.
This program converts EEG .edf files found in the source folder to log-mel spectrograms,
then creates 20-second frames with 10-second overlap from each channel's spectrogram.
Each frame is saved as a separate .npy file.

Added feature: Random masking of signal blocks per frame before spectrogram conversion for
robustness testing.

Workflow per channel:
    EEG signal -> normalize -> split into frames -> apply random masking -> convert to LMS -> save
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
    sample_rate = 16000  # Target sampling rate (will resample EEG to this)
    window_size = 400  # 25ms window at 16kHz
    n_fft = 400
    hop_size = 160  # 10ms stride at 16kHz
    n_mels = 80  # Number of mel frequency bins
    f_min = 0.5
    f_max = 100

    frame_duration = 20.0
    frame_overlap = 10.0
    frame_stride = frame_duration - frame_overlap


def apply_random_masking(signal_frame, sample_rate, min_mask_duration=0.25, max_mask_duration=1, num_masks=2):
    """
    Apply random masking to a single frame of signal
    """
    frame_masked = signal_frame.copy()
    frame_length = len(signal_frame)

    min_mask_samples = int(min_mask_duration * sample_rate)
    max_mask_samples = int(max_mask_duration * sample_rate)

    if min_mask_samples <= 0 or max_mask_samples <= 0:
        return frame_masked

    for i in range(num_masks):
        mask_length = np.random.randint(min_mask_samples, max_mask_samples + 1)
        max_start = max(0, frame_length - mask_length)
        if max_start <= 0:
            continue
        start_pos = np.random.randint(0, max_start + 1)
        end_pos = start_pos + mask_length
        frame_masked[start_pos:end_pos] = 0.0
        print(f"Mask {i}: start={start_pos}, end={end_pos}, length={mask_length}")

    return frame_masked


def split_into_frames(signal, frame_duration, frame_stride, sample_rate):
    """
    Split 1D signal into overlapping frames in samples
    """
    frame_length = int(frame_duration * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    frames = []

    start = 0
    while start + frame_length <= len(signal):
        frames.append(signal[start:start + frame_length])
        start += frame_step
    return frames


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

    def __call__(self, signal_frame):
        if isinstance(signal_frame, np.ndarray):
            signal_frame = torch.tensor(signal_frame, dtype=torch.float32)
        if signal_frame.dim() > 1:
            signal_frame = signal_frame.squeeze()
        x = self.to_spec(signal_frame)
        x = (x + torch.finfo(torch.float32).eps).log()
        return x


def _converter_worker(args):
    subpathname, from_dir, to_dir, prms, to_lms, suffix, min_length, verbose, masking_params = args
    from_dir_path_obj, to_dir_path_obj = Path(from_dir), Path(to_dir)
    edf_input_path = from_dir_path_obj / subpathname
    file_stem = Path(subpathname).stem
    relative_output_path = Path(subpathname).parent / file_stem
    folder_name = to_dir_path_obj / relative_output_path
    folder_name.mkdir(parents=True, exist_ok=True)

    try:
        raw = mne.io.read_raw_edf(str(edf_input_path), preload=True, verbose=False)
        eeg_data = raw.get_data()
        channel_names = raw.ch_names
        orig_sfreq = raw.info['sfreq']
        n_channels = eeg_data.shape[0]

        successful_channels = 0
        total_frames_saved = 0

        for channel_idx in range(n_channels):
            eeg_signal = eeg_data[channel_idx, :]

            # Resample if needed
            if orig_sfreq != prms.sample_rate:
                resample_ratio = prms.sample_rate / orig_sfreq
                n_samples_new = int(len(eeg_signal) * resample_ratio)
                eeg_signal = signal.resample(eeg_signal, n_samples_new)

            # Normalize
            eeg_signal = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)

            # Pad if too short
            min_samples = int(prms.sample_rate * prms.frame_duration)
            if len(eeg_signal) < min_samples:
                eeg_signal = np.pad(eeg_signal, (0, min_samples - len(eeg_signal)))

            # Split into frames
            frames = split_into_frames(eeg_signal, prms.frame_duration, prms.frame_stride, prms.sample_rate)

            # Apply masking per frame and convert to LMS
            channel_name = channel_names[channel_idx] if channel_idx < len(channel_names) else f"channel_{channel_idx:02d}"
            channel_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            channel_name = channel_name.replace(' ', '_')

            for frame_idx, frame in enumerate(frames):
                if masking_params['enable_masking']:
                    frame = apply_random_masking(
                        frame,
                        prms.sample_rate,
                        masking_params['min_mask_length'],
                        masking_params['max_mask_length'],
                        masking_params['num_masks']
                    )
                lms_frame = to_lms(frame)
                lms_frame = lms_frame[:, :,:2000]  # expected_time_frames = 2000
                print(lms_frame.shape)
                frame_filename = folder_name / f"{channel_name}_frame_{frame_idx:03d}.npy"
                np.save(frame_filename, lms_frame.numpy())
                total_frames_saved += 1

            successful_channels += 1

        if verbose:
            masking_status = "with masking" if masking_params['enable_masking'] else "without masking"
            print(f'Saved {total_frames_saved} frames from {successful_channels}/{n_channels} channels for {edf_input_path.name} {masking_status}')

    except Exception as e:
        print(f'ERROR processing {edf_input_path.name}: {e}')
        return f'{edf_input_path.name} (failed)'

    return f'{file_stem} ({successful_channels}/{n_channels} channels, {total_frames_saved} frames)'


def convert_eeg_batch(base_from_dir, base_to_dir,
                      start_subject_id=1, end_subject_id=109,
                      suffix='.edf', skip=0, min_length=20.0, verbose=False,
                      enable_masking=True, min_mask_length=0.25, max_mask_length=1, num_masks=2) -> None:

    base_from_path = Path(base_from_dir)
    base_to_path = Path(base_to_dir)
    all_generated_file_paths = []

    prms = EEG_FFT_parameters()
    to_lms = ToLogMelSpec(prms)
    masking_params = {
        'enable_masking': enable_masking,
        'min_mask_length': min_mask_length,
        'max_mask_length': max_mask_length,
        'num_masks': num_masks
    }

    for i in range(start_subject_id, end_subject_id + 1):
        subject_id = f"S{i:03d}"
        current_subject_from_dir = base_from_path / subject_id
        current_subject_to_dir = base_to_path / subject_id

        if not current_subject_from_dir.exists():
            print(f"WARNING: Source directory {current_subject_from_dir} not found. Skipping.")
            continue

        subject_edf_files = [f.relative_to(current_subject_from_dir) for f in current_subject_from_dir.glob(f'**/*{suffix}')]
        subject_edf_files = sorted(subject_edf_files)
        if skip > 0:
            subject_edf_files = subject_edf_files[skip:]

        if len(subject_edf_files) == 0:
            print(f"No {suffix} files found for {subject_id}.")
            continue

        worker_args_for_subject = [
            [f, str(current_subject_from_dir), str(current_subject_to_dir), prms, to_lms, suffix, min_length, verbose, masking_params]
            for f in subject_edf_files
        ]

        with Pool() as p:
            results = list(tqdm(p.imap(_converter_worker, worker_args_for_subject), total=len(worker_args_for_subject)))

        # Collect all .npy files for CSV
        for f in worker_args_for_subject:
            f_rel = Path(f[0]).stem
            output_folder = Path(f[2]) / Path(f[0]).parent / f_rel
            if output_folder.exists():
                for npy_file in output_folder.glob('*.npy'):
                    all_generated_file_paths.append(npy_file.relative_to(base_to_path))

    # Write CSV
    csv_filename = base_to_path / "files_audioset.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name'])
        for fp in all_generated_file_paths:
            writer.writerow([str(fp)])
    print(f"CSV saved: {csv_filename}. Total frames: {len(all_generated_file_paths)}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    convert_eeg_batch(
        "/app/1.0.0",
        "/app/data/masked_subset10_frames",
        start_subject_id=1,
        end_subject_id=10,
        verbose=True,
        enable_masking=True
    )
