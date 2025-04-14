import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter

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

def data_load_raw(path, frame_size=30, fs = 220, lowcut =0.5, highcut = 4.0):
  x_train=np.array([])
  y_train=[]
  for folder in (["TestingSet", "TestingSet_secret", "TrainingSet"]):
    for user in range(1,21):
      for session in range(1,2):
        for typ in (["fav", "same"]):
          filename = "user"+str(user)+"_"+typ+"_session"+str(session)+".csv"
          filepath = os.path.join(path, folder, filename)
          try:
            file = pd.read_csv(filepath)
            #data = np.array(file.iloc[:, 1:5])
            raw_data = np.array(file.iloc[:, 1:25])
            #data = np.array(file.iloc[:, [9,12,13,16]])
            data = np.array([])
            for chnl in range(raw_data.shape[0]):
              delta = butter_bandpass_filter(raw_data[chnl], lowcut, highcut, fs, order=3)
              if data.shape[0]==0:
                data=np.array([delta])
              else:
                data = np.concatenate((data,[delta]),axis=0)
            data = np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
            data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
            if data.shape[2]!=4:
              continue
            if x_train.shape[0]==0:
              x_train  = data
              y_train += [user-1]*data.shape[0]
            else:
              x_train  = np.concatenate((x_train,data), axis=0)
              y_train += [user-1]*data.shape[0]
            print(folder)
          except (FileNotFoundError, IndexError):
            continue
  print(x_train.shape)
  return x_train, np.array(y_train)
  
def data_load(path, frame_size=30):
  x_train=np.array([])
  y_train=[]
  for folder in (["TestingSet", "TestingSet_secret", "TrainingSet"]):
  #for folder in (["TrainingSet"]):
    for user in range(1,21):
      for session in range(1,6):
        for typ in (["fav", "same"]):
          filename = "user"+str(user)+"_"+typ+"_session"+str(session)+".csv"
          filepath = os.path.join(path, folder, filename)
          try:
            file = pd.read_csv(filepath)
            data = np.array(file.iloc[:, 1:25])
            #raw_data = np.array(file.iloc[:, 21:25])
            #data = np.array(file.iloc[:, [9,12,13,16]])
            data = np.lib.stride_tricks.sliding_window_view(data, (frame_size,data.shape[1]))[::frame_size//2, :]
            data=data.reshape(data.shape[0],data.shape[2],data.shape[3])
            if data.shape[2]!=24:
              continue
            if x_train.shape[0]==0:
              x_train  = data
              y_train += [user-1]*data.shape[0]
            else:
              x_train  = np.concatenate((x_train,data), axis=0)
              y_train += [user-1]*data.shape[0]
            print(folder, data.shape[0])
          except (FileNotFoundError, IndexError):
            continue
  print(x_train.shape)
  return x_train, np.array(y_train)
  
def norma(x_all):
  x = np.reshape(x_all,(x_all.shape[0]*x_all.shape[1],x_all.shape[2]))
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  x_all = np.reshape(x,(x_all.shape[0],x_all.shape[1],x_all.shape[2]))
  x=[]
  return x_all   