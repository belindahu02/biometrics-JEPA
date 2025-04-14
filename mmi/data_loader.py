import os
import mne
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def data_loader(path, num_class, sessions, frame_size=128, fs = 128.0, lowcut =0.5, highcut = 4.0):
    x_train = np.array([])
    y_train = []
    for subject in range(1, num_class+1):
      foldername = "S"+str(subject).rjust(3, '0')
      for sess in sessions:
        sessname = "R"+str(sess).rjust(2, '0')
        filename = os.path.join(path, foldername, foldername+sessname+".edf")
        data = mne.io.read_raw_edf(filename, verbose=False)
        raw_data = data.get_data()
        data = np.array([])
        for chnl in range(raw_data.shape[0]):
          delta = butter_bandpass_filter(raw_data[chnl], lowcut, highcut, fs, order=3)
          if data.shape[0]==0:
            data=np.array([delta])
          else:
            data = np.concatenate((data,[delta]),axis=0)
        data = data.T
        data=np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
        data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
        if x_train.shape[0]==0:
          x_train  = data
          y_train += [subject-1]*data.shape[0]
        else:
          x_train  = np.concatenate((x_train,data), axis=0)
          y_train += [subject-1]*data.shape[0]
    y_train = np.array(y_train)
    print(x_train.shape)
    return x_train, y_train
    
def process_data(path, num_class, sessions, frame_size=128, fs = 128.0):
    x_all = []
    y_all = []
    for subject in tqdm(range(1, num_class+1)):
        foldername = "S"+str(subject).rjust(3, '0')
        # for sess in range(1,num_sess+1):
        for sess in sessions:
            sessname = "R"+str(sess).rjust(2, '0')
            filename = os.path.join(path, foldername, foldername+sessname+".edf")
            data = mne.io.read_raw_edf(filename, verbose=False)
            raw_data = data.get_data()
            data = np.array([])
            # for chnl in channel_nums:
            for chnl in ([2, 11, 12, 17, 49, 59, 60, 63]):
                raw = raw_data[chnl]
                delta = butter_bandpass_filter(raw_data[chnl], 0.5, 4, fs, order=3)
                theta = butter_bandpass_filter(raw_data[chnl], 4, 8, fs, order=3)
                alpha = butter_bandpass_filter(raw_data[chnl], 8, 12, fs, order=3)
                beta = butter_bandpass_filter(raw_data[chnl], 12, 30, fs, order=3)
                all_waves = np.vstack((raw, delta, theta, alpha, beta))
                
                if data.shape[0]==0:
                    data = all_waves
                else:
                    data = np.vstack((data, all_waves))
            data = data.T
            data=np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
            data=data.reshape(data.shape[0],data.shape[2],data.shape[3])

            x_all.append(data)
            y_all += [subject-1]*data.shape[0]


    y_all = np.array(y_all)
    x_all = np.vstack(x_all)
    print(x_all.shape)
    print(y_all.shape)
    
    return x_all, y_all

def norma(x_all):
  x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  x_all = np.reshape(x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
  x=[]
  return x_all    