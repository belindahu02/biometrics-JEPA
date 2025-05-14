import os
import mne
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def load_and_filter_eeg(filename, fs, channels, bands):
    raw = mne.io.read_raw_edf(filename, verbose=False, preload=True)
    eeg = raw.get_data()

    data = []
    for ch in channels:
        raw_ch = eeg[ch]
        ch_bands = [butter_bandpass_filter(raw_ch, low, high, fs, order=3) for (low, high) in bands]
        ch_data = np.vstack([raw_ch] + ch_bands)
        data.append(ch_data)
    
    return np.vstack(data).T  # shape: (time, features)

def sliding_window(data, frame_size):
    try:
        windows = np.lib.stride_tricks.sliding_window_view(data, (frame_size, data.shape[1]))
        windows = windows[::frame_size // 2, :]
        return windows.reshape(windows.shape[0], windows.shape[2], windows.shape[3])
    except Exception as e:
        print(f"Sliding window error: {e}")
        return np.array([])

def process_data(path, num_class, sessions, frame_size=128, fs=128.0, channels=None, bands=None):
    x_all, y_all = [], []

    if channels is None:
        channels = [2, 11, 12, 17, 49, 59, 60, 63]

    if bands is None:
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]  # delta, theta, alpha, beta

    for subject in tqdm(range(1, num_class + 1), desc="Subjects"):
        subj_folder = f"S{subject:03d}"

        for sess in sessions:
            sess_name = f"R{sess:02d}"
            filepath = os.path.join(path, subj_folder, f"{subj_folder}{sess_name}.edf")
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue

            data = load_and_filter_eeg(filepath, fs, channels, bands)
            windowed = sliding_window(data, frame_size)

            if windowed.size > 0:
                x_all.append(windowed)
                y_all += [subject - 1] * windowed.shape[0]

    x_all = np.vstack(x_all)
    y_all = np.array(y_all)
    print(f"Final shape: {x_all.shape}, Labels: {y_all.shape}")
    return x_all, y_all

def normalize_data(x_all):
    flat = x_all.reshape(-1, x_all.shape[-1])
    scaler = StandardScaler()
    normed = scaler.fit_transform(flat)
    return normed.reshape(x_all.shape)
