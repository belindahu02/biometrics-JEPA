"""
EEG to log-mel spectrogram (LMS) converter.
This program converts EEG .edf files found in the source folder to log-mel spectrograms,
then stores them in the destination folder while holding the same relative path structure.
The conversion includes the following processes:
    - Multi-channel EEG to single channel (averaging or selecting specific channel)
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
    subpathname, from_dir, to_dir, prms, to_lms, suffix, min_length, channel_strategy, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir / (subpathname[:-len(suffix)] + '.npy')

    if to_name.exists():
        if verbose:
            print('already exist', subpathname)
        return ''

    # Load and convert EEG to log-mel spectrogram
    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(str(from_dir / subpathname), preload=True, verbose=False)

        # Get EEG data
        eeg_data = raw.get_data()  # Shape: (n_channels, n_samples)
        orig_sfreq = raw.info['sfreq']

        # Channel selection strategy
        if channel_strategy == 'average':
            # Average all channels
            eeg_signal = np.mean(eeg_data, axis=0)
        elif channel_strategy == 'first':
            # Use first channel
            eeg_signal = eeg_data[0, :]
        elif isinstance(channel_strategy, str) and channel_strategy.startswith('channel_'):
            # Use specific channel by index
            channel_idx = int(channel_strategy.split('_')[1])
            if channel_idx < eeg_data.shape[0]:
                eeg_signal = eeg_data[channel_idx, :]
            else:
                print(f'WARNING: Channel {channel_idx} not found, using first channel')
                eeg_signal = eeg_data[0, :]
        else:
            # Default to first channel
            eeg_signal = eeg_data[0, :]

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
                    print(f'Padding {subpathname} from {len(eeg_signal)} to {min_samples} samples')
                eeg_signal = np.pad(eeg_signal, (0, min_samples - len(eeg_signal)))

        # Convert to log-mel spectrogram
        lms = to_lms(eeg_signal)

    except Exception as e:
        print('ERROR failed to open or convert', subpathname, '-', str(e))
        return ''

    # Save the spectrogram
    to_name.parent.mkdir(parents=True, exist_ok=True)
    np.save(to_name, lms.numpy())

    if verbose:
        print(f'{from_dir / subpathname} -> {to_name}, shape: {lms.shape}')

    return to_name.name


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


def convert_eeg(from_dir, to_dir, suffix='.edf', skip=0, min_length=6.1,
                channel_strategy='average', verbose=False) -> None:
    """
    Convert EEG files to log-mel spectrograms.

    Args:
        from_dir: Source directory containing .edf files
        to_dir: Destination directory for .npy files
        suffix: File extension to process (default: '.edf')
        skip: Number of files to skip (default: 0)
        min_length: Minimum length in seconds (default: 6.1)
        channel_strategy: How to handle multiple EEG channels:
            - 'average': Average all channels
            - 'first': Use first channel only
            - 'channel_N': Use specific channel N (e.g., 'channel_0', 'channel_1')
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
    print(f'Channel strategy: {channel_strategy}')
    print(f'Output shape will be: (80, time_frames)')
    print(f'Frequency range: {prms.f_min}-{prms.f_max} Hz')

    # Process files
    with Pool() as p:
        args = [[f, from_dir, to_dir, prms, to_lms, suffix, min_length, channel_strategy, verbose]
                for f in files]
        results = list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    successful = [r for r in results if r]
    print(f'Finished. Successfully converted {len(successful)}/{len(files)} files.')


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # fire.Fire(convert_eeg)

    # test with a few samples S001
    convert_eeg("/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0/S001", "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/S001")
