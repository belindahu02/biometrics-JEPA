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
    from_dir, to_dir = Path(from_dir), Path(to_dir)

    # Create folder name from file name (without extension)
    file_stem = Path(subpathname).stem  # e.g., "S001R01" from "S001R01.edf"
    folder_name = to_dir / file_stem

    # Check if folder already exists and has files
    if folder_name.exists() and any(folder_name.glob('*.npy')):
        if verbose:
            print(f'Folder {file_stem} already exists with files')
        return f'{file_stem} (already exists)'

    # Load and convert EEG to log-mel spectrogram
    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(str(from_dir / subpathname), preload=True, verbose=False)

        # Get EEG data and channel names
        eeg_data = raw.get_data()  # Shape: (n_channels, n_samples)
        channel_names = raw.ch_names
        orig_sfreq = raw.info['sfreq']
        n_channels = eeg_data.shape[0]

        if verbose:
            print(f'Processing {subpathname}: {n_channels} channels, {len(channel_names)} channel names')

        # Create output folder
        folder_name.mkdir(parents=True, exist_ok=True)

        # Process each channel separately
        successful_channels = []
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

                # Save the spectrogram for this channel
                channel_filename = folder_name / f"{channel_name}.npy"
                np.save(channel_filename, lms.numpy())
                successful_channels.append(channel_name)

                if verbose:
                    print(f'  Channel {channel_idx} ({channel_name}) -> {channel_filename}, shape: {lms.shape}')

            except Exception as e:
                print(f'ERROR processing channel {channel_idx} in {subpathname}: {str(e)}')
                continue

        # Save channel information
        channel_info = {
            'original_file': subpathname,
            'original_sfreq': orig_sfreq,
            'target_sfreq': prms.sample_rate,
            'n_channels': n_channels,
            'channel_names': channel_names,
            'successful_channels': successful_channels,
            'spectrogram_shape': lms.shape if 'lms' in locals() else None
        }

        # Save metadata
        # metadata_file = folder_name / "metadata.npy"
        # np.save(metadata_file, channel_info)

        if verbose:
            print(f'Saved {len(successful_channels)}/{n_channels} channels for {subpathname}')

    except Exception as e:
        print('ERROR failed to open or convert', subpathname, '-', str(e))
        return f'{subpathname} (failed)'

    return f'{file_stem} ({len(successful_channels)}/{n_channels} channels)'


class ToLogMelSpec:
    def __init__(self, cfg):
        # Spectrogram extractor - same as audio converter but adapted for EEG frequency range
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
        # Convert to tensor and add batch dimension if needed
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)

        # Ensure signal is 1D
        if signal.dim() > 1:
            signal = signal.squeeze()

        # Compute mel spectrogram
        x = self.to_spec(signal)

        # Convert to log scale (add small epsilon to avoid log(0))
        x = (x + torch.finfo(torch.float32).eps).log()

        return x


def convert_eeg(from_dir, to_dir, suffix='.edf', skip=0, min_length=6.1, verbose=False) -> None:
    """
    Convert EEG files to log-mel spectrograms, processing all channels separately.

    Args:
        from_dir: Source directory containing .edf files
        to_dir: Destination directory for .npy files (each file will create a subfolder)
        suffix: File extension to process (default: '.edf')
        skip: Number of files to skip (default: 0)
        min_length: Minimum length in seconds (default: 6.1)
        verbose: Print detailed progress information
    """
    from_dir = str(from_dir)
    files = [str(f).replace(from_dir, '') for f in Path(from_dir).glob(f'**/*{suffix}')]
    files = [f[1:] if f[0] == '/' else f for f in files]
    files = sorted(files)

    if skip > 0:
        files = files[skip:]

    if len(files) == 0:
        print(f'No {suffix} files found in {from_dir}')
        return

    prms = EEG_FFT_parameters()
    to_lms = ToLogMelSpec(prms)

    print(f'Processing {len(files)} {suffix} files...')
    print(f'Target sampling rate: {prms.sample_rate} Hz')
    print(f'Processing ALL channels separately')
    print(f'Output shape per channel: (80, time_frames)')
    print(f'Frequency range: {prms.f_min}-{prms.f_max} Hz')

    # Process files
    with Pool() as p:
        args = [[f, from_dir, to_dir, prms, to_lms, suffix, min_length, verbose]
                for f in files]
        results = list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    successful = [r for r in results if r and 'failed' not in r]
    print(f'Finished. Successfully processed {len(successful)}/{len(files)} files.')

    # Print summary
    for result in results:
        if result:
            print(f'  {result}')


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # fire.Fire(convert_eeg)

    for i in range(1, 109):  # 1 to 109 inclusive
        subject_id = f"S{i:03d}"  # Format as S001, S002, etc.
        source_path = f"/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0/{subject_id}"
        dest_path = f"/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/{subject_id}"

        convert_eeg(source_path, dest_path)