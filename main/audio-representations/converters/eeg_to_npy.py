import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from scipy.signal import spectrogram
import os
from typing import Optional, Tuple, Union


def eeg_to_spectrogram(
        edf_file_path: str,
        channel_index: int = 2,
        output_path: str = "spectrogram.npy",
        sample_length: Optional[int] = 7500,
        sampling_rate: int = 125,
        noverlap: int = 1,
        return_raw_data: bool = False,
        save_format: str = "npy",
        figsize: Tuple[int, int] = (2.24, 2.24),
        dpi: int = 100,
        normalize: bool = True,  # New parameter for normalization
        log_scale: bool = True  # New parameter for log scaling
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert EEG signal from EDF file to spectrogram data (NumPy array or PNG image).
    Modified to match the format expected by audio models.

    Args:
        edf_file_path (str): Path to the EDF file
        channel_index (int): Index of the EEG channel to process (default: 2)
        output_path (str): Path where spectrogram will be saved
        sample_length (int, optional): Number of samples to process. If None, use entire signal
        sampling_rate (int): Sampling frequency in Hz (default: 125)
        noverlap (int): Number of points to overlap between segments (default: 1)
        return_raw_data (bool): Whether to return raw spectrogram data (default: False)
        save_format (str): Format to save - "npy" for NumPy array or "png" for image (default: "npy")
        figsize (tuple): Figure size for the spectrogram (default: (2.24, 2.24))
        dpi (int): Resolution of the saved image (default: 100)
        normalize (bool): Whether to normalize the spectrogram (default: True)
        log_scale (bool): Whether to apply log scaling like audio spectrograms (default: True)

    Returns:
        None or tuple: If return_raw_data is True, returns (frequencies, times, Sxx) from scipy spectrogram

    Raises:
        FileNotFoundError: If EDF file doesn't exist
        ValueError: If channel index is invalid or save_format is invalid
        Exception: For other processing errors
    """

    # Validate input file
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")

    # Validate save format
    if save_format not in ["npy", "png"]:
        raise ValueError(f"Invalid save_format: {save_format}. Must be 'npy' or 'png'")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw = None
    try:
        # Read EDF file with MNE
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

        # Validate channel index
        if channel_index >= raw.info['nchan'] or channel_index < 0:
            raise ValueError(
                f"Invalid channel index {channel_index}. File has {raw.info['nchan']} channels (0-{raw.info['nchan'] - 1})")

        # Read signal data
        signal_data = raw.get_data()[channel_index]
        print(f"Original signal shape: {signal_data.shape}")

        # Determine sample length
        if sample_length is None:
            sample_length = len(signal_data)
        else:
            sample_length = min(sample_length, len(signal_data))

        # Extract the portion of signal to process
        signal_segment = signal_data[:sample_length]
        print(f"Processing {sample_length} samples")

        # Generate spectrogram using scipy
        frequencies, times, Sxx = spectrogram(signal_segment, fs=sampling_rate, noverlap=noverlap)

        if save_format == "npy":
            # Process spectrogram to match audio format
            if log_scale:
                # Apply log scaling similar to audio spectrograms
                # Add small epsilon to avoid log(0)
                Sxx = np.log(Sxx + np.finfo(float).eps)

            if normalize:
                # Normalize the spectrogram
                Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + np.finfo(float).eps)

            # Reshape to match expected format: [frequency_bins, time_bins]
            # This matches the audio format where spectrograms are 2D
            processed_spectrogram = Sxx

            print(f"Original spectrogram shape: {Sxx.shape}")
            print(f"Processed spectrogram shape: {processed_spectrogram.shape}")

            # Save the processed spectrogram
            np.save(output_path, processed_spectrogram)

            print(f"Spectrogram data saved to: {output_path}")
            print(f"Final shape: {processed_spectrogram.shape}")

        elif save_format == "png":
            # Create the spectrogram plot (original behavior)
            plt.figure(figsize=figsize, dpi=dpi)

            # Use matplotlib's specgram for visualization
            im = specgram(signal_segment, Fs=sampling_rate, noverlap=noverlap)[3]

            # Remove all axes, labels, and whitespace
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())

            # Save the clean spectrogram image
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close()  # Close the figure to free memory

            print(f"Spectrogram image saved to: {output_path}")

        # Return raw data if requested
        if return_raw_data:
            return frequencies, times, Sxx

    except Exception as e:
        if save_format == "png":
            plt.close('all')  # Clean up any open figures
        print(f"Error processing EEG file: {str(e)}")
        raise

    finally:
        # Clean up (MNE handles cleanup automatically)
        pass


def batch_eeg_to_spectrogram(
        edf_files: list,
        output_directory: str = "spectrograms",
        save_format: str = "npy",
        **kwargs
) -> list:
    """
    Process multiple EDF files to spectrograms in batch.

    Args:
        edf_files (list): List of EDF file paths
        output_directory (str): Directory to save all spectrograms
        save_format (str): Format to save - "npy" or "png" (default: "npy")
        **kwargs: Additional arguments to pass to eeg_to_spectrogram

    Returns:
        list: List of successfully processed files
    """

    # Create format-specific subdirectory
    format_output_dir = os.path.join(output_directory, save_format)
    if not os.path.exists(format_output_dir):
        os.makedirs(format_output_dir)

    successful_files = []

    for i, edf_file in enumerate(edf_files):
        try:
            # Generate output filename with appropriate extension
            base_name = os.path.splitext(os.path.basename(edf_file))[0]
            extension = ".npy" if save_format == "npy" else ".png"
            output_path = os.path.join(format_output_dir, f"{base_name}_spectrogram{extension}")

            # Process the file
            eeg_to_spectrogram(edf_file, output_path=output_path, save_format=save_format, **kwargs)
            successful_files.append(edf_file)

        except Exception as e:
            print(f"Failed to process {edf_file}: {str(e)}")

    print(f"Successfully processed {len(successful_files)}/{len(edf_files)} files")
    return successful_files


def process_subject_folders(
        base_dataset_path: str,
        subject_pattern: str = "S???",
        file_pattern: str = "*.edf",
        output_directory: str = "spectrograms",
        save_format: str = "npy",
        **kwargs
) -> dict:
    """
    Process EDF files from multiple subject folders (S001, S002, etc.).

    Args:
        base_dataset_path (str): Path to the dataset directory containing subject folders
        subject_pattern (str): Pattern to match subject folders (default: "S???")
        file_pattern (str): Pattern to match EDF files within each subject folder (default: "*.edf")
        output_directory (str): Directory to save all spectrograms
        save_format (str): Format to save - "npy" or "png" (default: "npy")
        **kwargs: Additional arguments to pass to eeg_to_spectrogram

    Returns:
        dict: Dictionary with subject IDs as keys and lists of processed files as values
    """

    import glob

    if not os.path.exists(base_dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {base_dataset_path}")

    # Find all subject folders
    subject_folders = glob.glob(os.path.join(base_dataset_path, subject_pattern))
    subject_folders.sort()  # Sort to process in order

    if not subject_folders:
        print(f"No subject folders found matching pattern '{subject_pattern}' in {base_dataset_path}")
        return {}

    print(f"Found {len(subject_folders)} subject folders")

    # Create format-specific output directory
    format_output_dir = os.path.join(output_directory, save_format)
    if not os.path.exists(format_output_dir):
        os.makedirs(format_output_dir)

    results = {}

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        print(f"\nProcessing subject: {subject_id}")

        # Find all EDF files in this subject folder
        edf_files = glob.glob(os.path.join(subject_folder, file_pattern))
        edf_files.sort()

        if not edf_files:
            print(f"  No EDF files found in {subject_folder}")
            results[subject_id] = []
            continue

        print(f"  Found {len(edf_files)} EDF files")

        # Create subject-specific output directory within format folder
        subject_output_dir = os.path.join(format_output_dir, subject_id)
        if not os.path.exists(subject_output_dir):
            os.makedirs(subject_output_dir)

        successful_files = []

        for edf_file in edf_files:
            try:
                # Generate output filename with appropriate extension
                base_name = os.path.splitext(os.path.basename(edf_file))[0]
                extension = ".npy" if save_format == "npy" else ".png"
                output_path = os.path.join(subject_output_dir, f"{base_name}_spectrogram{extension}")

                # Process the file
                eeg_to_spectrogram(edf_file, output_path=output_path, save_format=save_format, **kwargs)
                successful_files.append(edf_file)
                print(f"    ✓ Processed: {base_name}")

            except Exception as e:
                print(f"    ✗ Failed to process {os.path.basename(edf_file)}: {str(e)}")

        results[subject_id] = successful_files
        print(f"  Subject {subject_id}: {len(successful_files)}/{len(edf_files)} files processed")

    total_processed = sum(len(files) for files in results.values())
    total_files = sum(len(glob.glob(os.path.join(folder, file_pattern))) for folder in subject_folders)
    print(f"\nOverall: {total_processed}/{total_files} files processed successfully")
    print(f"Files saved to: {format_output_dir}")

    return results


def load_spectrogram_npy(npy_file_path: str) -> np.ndarray:
    """
    Load a spectrogram from a .npy file.

    Args:
        npy_file_path (str): Path to the .npy file

    Returns:
        np.ndarray: The spectrogram data
    """
    if not os.path.exists(npy_file_path):
        raise FileNotFoundError(f"NPY file not found: {npy_file_path}")

    return np.load(npy_file_path)


def get_edf_files_from_subjects(
        base_dataset_path: str,
        subject_ids: list = None,
        file_pattern: str = "*.edf"
) -> list:
    """
    Get list of EDF files from specific subject folders.

    Args:
        base_dataset_path (str): Path to the dataset directory
        subject_ids (list): List of subject IDs (e.g., ['S001', 'S002']). If None, gets all subjects.
        file_pattern (str): Pattern to match EDF files (default: "*.edf")

    Returns:
        list: List of EDF file paths
    """

    import glob

    if subject_ids is None:
        # Get all subject folders
        subject_folders = glob.glob(os.path.join(base_dataset_path, "S???"))
    else:
        # Get specific subject folders
        subject_folders = [os.path.join(base_dataset_path, subject_id) for subject_id in subject_ids]

    all_edf_files = []

    for subject_folder in subject_folders:
        if os.path.exists(subject_folder):
            edf_files = glob.glob(os.path.join(subject_folder, file_pattern))
            all_edf_files.extend(edf_files)

    return sorted(all_edf_files)


# Additional utility function to check spectrogram dimensions
def check_spectrogram_compatibility(eeg_spectrogram_path: str, audio_spectrogram_path: str = None):
    """
    Check if EEG spectrogram dimensions are compatible with audio model expectations.

    Args:
        eeg_spectrogram_path (str): Path to EEG spectrogram .npy file
        audio_spectrogram_path (str, optional): Path to audio spectrogram for comparison
    """

    eeg_spec = load_spectrogram_npy(eeg_spectrogram_path)
    print(f"EEG Spectrogram shape: {eeg_spec.shape}")
    print(f"EEG Spectrogram dtype: {eeg_spec.dtype}")
    print(f"EEG Spectrogram range: [{eeg_spec.min():.4f}, {eeg_spec.max():.4f}]")

    if audio_spectrogram_path and os.path.exists(audio_spectrogram_path):
        audio_spec = np.load(audio_spectrogram_path)
        print(f"\nAudio Spectrogram shape: {audio_spec.shape}")
        print(f"Audio Spectrogram dtype: {audio_spec.dtype}")
        print(f"Audio Spectrogram range: [{audio_spec.min():.4f}, {audio_spec.max():.4f}]")

        print(f"\nDimension compatibility:")
        print(f"  Same number of dimensions: {len(eeg_spec.shape) == len(audio_spec.shape)}")
        if len(eeg_spec.shape) == len(audio_spec.shape):
            for i, (eeg_dim, audio_dim) in enumerate(zip(eeg_spec.shape, audio_spec.shape)):
                print(f"  Dimension {i}: EEG={eeg_dim}, Audio={audio_dim}, Match={eeg_dim == audio_dim}")


# Example usage
if __name__ == "__main__":
    try:
        dataset_path = "../../../mmi/dataset/physionet.org/files/eegmmidb/1.0.0"

        # Process to .npy files with audio-compatible format
        process_subject_folders(
            dataset_path,
            subject_pattern="S001",  # S001-S009
            save_format="npy",
            output_directory="test",
            normalize=True,  # Apply normalization
            log_scale=True  # Apply log scaling like audio
        )

        # Example of checking compatibility
        # check_spectrogram_compatibility("datasets/npy/S001/S001R01_spectrogram.npy")

    except Exception as e:
        print(f"Error: {e}")