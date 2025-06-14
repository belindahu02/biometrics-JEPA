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
    output_path: str = "spectrogram.png",
    sample_length: Optional[int] = 7500,
    sampling_rate: int = 125,
    noverlap: int = 1,
    return_raw_data: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100
) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert EEG signal from EDF file to spectrogram image.
    
    Args:
        edf_file_path (str): Path to the EDF file
        channel_index (int): Index of the EEG channel to process (default: 2)
        output_path (str): Path where spectrogram image will be saved
        sample_length (int, optional): Number of samples to process. If None, use entire signal
        sampling_rate (int): Sampling frequency in Hz (default: 125)
        noverlap (int): Number of points to overlap between segments (default: 1)
        return_raw_data (bool): Whether to return raw spectrogram data (default: False)
        figsize (tuple): Figure size for the spectrogram (default: (10, 6))
        dpi (int): Resolution of the saved image (default: 100)
    
    Returns:
        None or tuple: If return_raw_data is True, returns (frequencies, times, Sxx) from scipy spectrogram
    
    Raises:
        FileNotFoundError: If EDF file doesn't exist
        ValueError: If channel index is invalid
        Exception: For other processing errors
    """
    
    # Validate input file
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    
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
            raise ValueError(f"Invalid channel index {channel_index}. File has {raw.info['nchan']} channels (0-{raw.info['nchan']-1})")
        
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
        
        # Generate spectrogram using scipy (for raw data if needed)
        frequencies, times, Sxx = spectrogram(signal_segment, fs=sampling_rate, noverlap=noverlap)
        
        # Create the spectrogram plot
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
        
        print(f"Spectrogram saved to: {output_path}")
        
        # Return raw data if requested
        if return_raw_data:
            return frequencies, times, Sxx
            
    except Exception as e:
        plt.close('all')  # Clean up any open figures
        print(f"Error processing EEG file: {str(e)}")
        raise
        
    finally:
        # Clean up (MNE handles cleanup automatically)
        pass


def batch_eeg_to_spectrogram(
    edf_files: list,
    output_directory: str = "spectrograms",
    **kwargs
) -> list:
    """
    Process multiple EDF files to spectrograms in batch.
    
    Args:
        edf_files (list): List of EDF file paths
        output_directory (str): Directory to save all spectrograms
        **kwargs: Additional arguments to pass to eeg_to_spectrogram
    
    Returns:
        list: List of successfully processed files
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    successful_files = []
    
    for i, edf_file in enumerate(edf_files):
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(edf_file))[0]
            output_path = os.path.join(output_directory, f"{base_name}_spectrogram.png")
            
            # Process the file
            eeg_to_spectrogram(edf_file, output_path=output_path, **kwargs)
            successful_files.append(edf_file)
            
        except Exception as e:
            print(f"Failed to process {edf_file}: {str(e)}")
    
    print(f"Successfully processed {len(successful_files)}/{len(edf_files)} files")
    return successful_files


# Example usage
if __name__ == "__main__":
    # Single file processing
    try:
        eeg_to_spectrogram(
            edf_file_path="/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
            channel_index=2,
            output_path="output/spectrogram.png",
            sample_length=7500,
            sampling_rate=125
        )
    except Exception as e:
        print(f"Error: {e}")
    
    # Batch processing example
    # edf_files = ["file1.edf", "file2.edf", "file3.edf"]
    # batch_eeg_to_spectrogram(edf_files, output_directory="batch_spectrograms")
    
    # Example with raw data return
    # freq, times, raw_spec = eeg_to_spectrogram(
    #     "data/shhs1-200001.edf",
    #     return_raw_data=True
    # )