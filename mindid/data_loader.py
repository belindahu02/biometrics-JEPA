
import scipy.io as sc
import numpy as np
from sklearn import preprocessing
from scipy.signal import butter, lfilter

def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

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

def load_data(len_sample=1, full=7000, mat_file="EID-S.mat", feature_key="eeg_close_8sub_1file", n_fea=14):
    len_a = full // len_sample
    labels = [np.ones(len_a) * i for i in range(8)]
    label = np.hstack(labels).reshape(-1, 1)

    feature = sc.loadmat(mat_file)
    all_data = feature[feature_key]
    all_data = all_data[0:full*8, 0:n_fea]

    data_f1 = []
    fs = 128.0
    lowcut = 0.5
    highcut = 4.0
    for i in range(all_data.shape[1]):
        x = all_data[:, i]
        y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
        data_f1.append(y)

    data_f1 = np.array(data_f1).T
    all_data = data_f1

    all_data = np.hstack((all_data, label))
    data_size = all_data.shape[0]
    feature_all = all_data[:, 0:n_fea] - 4200
    feature_all = preprocessing.scale(feature_all)
    label_all = all_data[:, n_fea:n_fea+1]
    all_data = np.hstack((feature_all, label_all))

    np.random.shuffle(all_data)
    train_data = all_data[0:int(data_size * 0.875)]
    test_data = all_data[int(data_size * 0.875):]

    feature_training = train_data[:, 0:n_fea].reshape([train_data.shape[0], len_sample, n_fea])
    feature_testing = test_data[:, 0:n_fea].reshape([test_data.shape[0], len_sample, n_fea])

    label_training = one_hot(train_data[:, n_fea])
    label_testing = one_hot(test_data[:, n_fea])

    batch_size = int(data_size * 0.125)
    n_group = 7
    train_fea = [feature_training[i * batch_size:(i + 1) * batch_size] for i in range(n_group)]
    train_label = [label_training[i * batch_size:(i + 1) * batch_size] for i in range(n_group)]

    return train_fea, train_label, feature_testing, label_testing, n_fea, len_sample, batch_size, n_group

# EEG-S dataset is a subset of EEG_ID_label6.mat. 1 trial, 7000 samples per sub

# feature = sc.loadmat("/home/xiangzhang/matlabwork/eegmmidb/EEG_ID_label6.mat")  # 1trial, 13500 samples each subject
# all = feature['EEG_ID_label6']
# n_fea = 64
# all = all[0:21000*8, 0:n_fea]
# print all.shape
#
# a1 = all[0:7000]  # select 7000 samples from 135000
# for i in range(2,9):
#     b = all[13500*(i-1):13500*i]
#     c = b[0:7000]
#     print c.shape
#     a1 = np.vstack((a1, c))
#     print i, a1.shape
#
# all = a1
# print all.shape

# def load_eeg_s_subset(mat_path: str, n_subjects: int = 8, samples_per_subject: int = 7000, total_samples_per_subject: int = 13500, n_features: int = 64):
#     """
#     Loads a subset of the EEG_ID_label6.mat dataset to form the EEG-S dataset.
    
#     Args:
#         mat_path (str): Path to the .mat file.
#         n_subjects (int): Number of subjects to extract data from.
#         samples_per_subject (int): Number of samples to keep per subject.
#         total_samples_per_subject (int): Total available samples per subject in the original file.
#         n_features (int): Number of features (channels).

#     Returns:
#         np.ndarray: The EEG-S dataset with shape (n_subjects * samples_per_subject, n_features)
#     """
#     mat = sc.loadmat(mat_path)
#     all_data = mat['EEG_ID_label6'][:n_subjects * total_samples_per_subject, :n_features]
    
#     eeg_s_subset = all_data[0:samples_per_subject]
#     for i in range(2, n_subjects + 1):
#         subject_data = all_data[total_samples_per_subject * (i - 1):total_samples_per_subject * i]
#         subject_subset = subject_data[:samples_per_subject]
#         eeg_s_subset = np.vstack((eeg_s_subset, subject_subset))
    
#     return eeg_s_subset
