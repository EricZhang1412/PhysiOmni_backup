import os
import scipy.io
import numpy as np
import mne
from scipy.interpolate import interp1d
import pickle

dataset_base = './FBM/raw/'
channel_names = {}

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'T1', 'T2', 'I1', 'I2', 'HEO', 'VEO', 'ECG', 'EMG', \
    'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'P3-O1', 'P4-O2', 'T3-T5', 'T4-T6', \
    'C3-P3', 'C4-A1', 'C4-P4', 'F2-F4'
]

EEG_order = ['FP1', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', \
               'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', \
                'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', \
                'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', \
                'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', \
                'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'
            ]
joint_names = ['jL5S1', 'jL4L3', 'jL1T12', 'jT9T8', 'jT1C7', 'jC1Head', \
                    'jRightC7Shoulder', 'jRightShoulder', 'jRightElbow', 'jRightWrist', \
                    'jLeftC7Shoulder', 'jLeftShoulder', 'jLeftElbow', 'jLeftWrist', \
                    'jRightHip', 'jRightKnee', 'jRightAnkle', 'jRightBallFoot', \
                    'jLeftHip', 'jLeftKnee', 'jLeftAnkle', 'jLeftBallFoot']
EOG_channels = ['VEO', 'HEO']
for dim in ['X', 'Y', 'Z']:
    for name in joint_names:
        standard_1020.append(name + dim)
eeg_l_freq = 0.1
eeg_h_freq = 75.0
eeg_rsfreq = 200
eeg_samp_rate = 1000

emg_l_freq = 5
emg_h_freq = 200
emg_rsfreq = 500
emg_samp_rate = 1000

eog_l_freq = 0.1
eog_h_freq = 75.0
eog_rsfreq = 200
eog_samp_rate = 1000

kin_rsfreq = 20

seg_freq = 20
window_time = 2

for file in os.listdir(dataset_base):
    if not file.endswith('.bvct'):
        continue
    sbj = file.split('-')[0].split('_')[-1]
    channel_names[sbj] = []
    with open(dataset_base + file, 'r') as f:
        lines = f.readlines()[:640]
        for i in range(len(lines)):
            lines[i] = lines[i].replace(' ', ',')
            if not '<Name>' in lines[i]:
                continue
            channel_name = lines[i].split('<Name>')[1].split('</Name>')[0]
            if (not channel_name.upper() in standard_1020) or channel_name.upper() == 'A1' or channel_name.upper() == 'A2':
                # print(channel_name.upper())
                continue
            if channel_name.upper() == 'FT9':
                channel_name = 'AFZ'
            if channel_name.upper() == 'FT10':
                channel_name = 'FCZ'
            channel_names[sbj].append(channel_name.upper())

info = mne.create_info(ch_names=EEG_order, sfreq=1000, ch_types='eeg')    


def sample_generation(data, samp_freq, seg_freq, t):
    C, T = data.shape

    sample_length = samp_freq * t

    num_samples = (T - sample_length ) // int(samp_freq / seg_freq) + 1

    samples = np.zeros((num_samples, C, sample_length))

    for i in range(num_samples):
        start_idx = int(i * (samp_freq / seg_freq))
        end_idx = start_idx + sample_length

        samples[i] = data[:, start_idx:end_idx]

    return samples


def preprocessing_EEG(eeg_file, eeg_l_freq=0.1, eeg_h_freq=75.0, eeg_rsfreq=200, eeg_samp_rate=1000, seg_freq=10, t=2):
    mat = scipy.io.loadmat(dataset_base + eeg_file)
    eeg_data = mat['eeg'][0][0]['rawdata']  # Adjust the key based on the actual structure of your .mat file
    channel_order = channel_names[sbj]
    ordered_data = []
    for channel in standard_1020:
        if channel in channel_order:
            idx = channel_order.index(channel)
            ordered_data.append(eeg_data[idx])
    eeg_data = np.array(ordered_data)
    eeg_mne_data = mne.io.RawArray(eeg_data, info)
    # Band-pass filter the EEG signal between 0.1Hz and 75Hz
    eeg_mne_data.filter(l_freq=eeg_l_freq, h_freq=eeg_h_freq, method='iir')
    # Apply a notch filter at 50Hz to remove power line noise
    eeg_mne_data.notch_filter(freqs=50, method='iir')
    # Resample the EEG data
    eeg_mne_data.resample(eeg_rsfreq, npad="auto")

    eeg_data = eeg_mne_data.get_data()
    del eeg_mne_data
    eeg_samples = sample_generation(eeg_data, eeg_rsfreq, seg_freq, 2)
    return eeg_samples


def preprocessing_EMG(emg_file, emg_l_freq=5, emg_h_freq=200, emg_rsfreq=500, emg_samp_rate=1000, seg_freq=10, t=2):
    emg_data = scipy.io.loadmat(dataset_base + emg_file)
    emg_data = np.concatenate((emg_data['emg']['left'][0][0], emg_data['emg']['right'][0][0]), axis=0)
    # Band-pass filter the EMG signal
    emg_data = mne.filter.filter_data(emg_data, sfreq=emg_samp_rate, l_freq=emg_l_freq, h_freq=emg_h_freq, method='iir')
    # Apply a notch filter at 50Hz to remove power line noise
    emg_data = mne.filter.notch_filter(emg_data, Fs=emg_samp_rate, freqs=50, method='iir')
    emg_data = mne.filter.notch_filter(emg_data, Fs=emg_samp_rate, freqs=100, method='iir')
    emg_data = mne.filter.notch_filter(emg_data, Fs=emg_samp_rate, freqs=150, method='iir')
    emg_data = mne.filter.notch_filter(emg_data, Fs=emg_samp_rate, freqs=200, method='iir')
    
    num_samples = emg_data.shape[1]
    time_original = np.linspace(0, num_samples / emg_samp_rate, num_samples)
    num_new_samples = int(num_samples * emg_rsfreq / emg_samp_rate)
    time_new = np.linspace(0, num_samples / emg_samp_rate, num_new_samples)
    emg_data_resampled = np.zeros((emg_data.shape[0], num_new_samples))
    for i in range(emg_data.shape[0]):
        interp_func = interp1d(time_original, emg_data[i, :], kind='linear', fill_value="extrapolate")
        emg_data_resampled[i, :] = interp_func(time_new)
    emg_data = emg_data_resampled
    del emg_data_resampled
    emg_samples= sample_generation(emg_data, emg_rsfreq, seg_freq, t)
    return emg_samples


def preprocessing_EOG(eog_file, eog_l_freq=0.1, eog_h_freq=75.0, eog_rsfreq=200, eog_samp_rate=1000, seg_freq=10, t=2):
    mat = scipy.io.loadmat(dataset_base + eog_file)
    eog_data = mat['eeg'][0][0]['eogdata']
    eog_data = mne.filter.filter_data(eog_data, sfreq=eog_samp_rate, l_freq=eog_l_freq, h_freq=eog_h_freq, method='iir')
    eog_data = mne.filter.notch_filter(eog_data, Fs=eog_samp_rate, freqs=50, method='iir')
    eog_data_resampled = np.zeros((eog_data.shape[0], eog_data.shape[1] * eog_rsfreq // eog_samp_rate))
    for i in range(eog_data.shape[0]):
        interp_func = interp1d(np.linspace(0, eog_data.shape[1] / eog_samp_rate, eog_data.shape[1]), eog_data[i, :], kind='linear', fill_value="extrapolate")
        eog_data_resampled[i, :] = interp_func(np.linspace(0, eog_data.shape[1] / eog_samp_rate, eog_data.shape[1] * eog_rsfreq // eog_samp_rate))
    eog_data_compress = np.zeros((2, eog_data_resampled.shape[1]))
    eog_data_compress[0, :] = eog_data_resampled[2, :] - eog_data_resampled[3, :]
    eog_data_compress[1, :] = eog_data_resampled[1, :] - eog_data_resampled[0, :]
    del eog_data_resampled
    eog_samples = sample_generation(eog_data_compress, eog_rsfreq, seg_freq, t)
    return eog_samples


def preprocessing_KIN(kin_file, kin_rsfreq=10, seg_freq=10, t=2):
    kin_data = scipy.io.loadmat(dataset_base + kin_file)['kin']['data'][0][0][0][0]['jointAngle']
    print([name for name in kin_data.dtype.names])
    kin_data = np.concatenate(([np.array(kin_data[name][0][0]) for name in joint_names]), axis=1)
    kin_samp_rate = int(scipy.io.loadmat(dataset_base + kin_file)['kin']['srate'])
    T = kin_data.shape[0]
    time_original = np.linspace(0, T / kin_samp_rate, T).flatten()
    num_new_samples = int(T * kin_rsfreq / kin_samp_rate)
    time_new = np.linspace(0, T / kin_samp_rate, num_new_samples)
    kin_data_resampled = np.zeros((num_new_samples, kin_data.shape[1]))
    for i in range(kin_data.shape[1]):
        y = kin_data[:, i].flatten()
        interp_func = interp1d(time_original, y, kind='linear', fill_value="extrapolate")
        kin_data_resampled[:, i] = interp_func(time_new)
    kin_data = kin_data_resampled.T
    del kin_data_resampled
    kin_samples = sample_generation(kin_data, kin_rsfreq, seg_freq, 2)
    label = kin_samples[:, :, -1]
    return label

for file in os.listdir(dataset_base):
    if not file.endswith('-eeg.mat'):
        continue

    trial = file.split('-')[1][1:]
    if trial == '00':
        continue
    sbj = file.split('-')[0].split('_')[-1]
    if sbj !='02':
        continue

    emg_file = file.replace('-eeg.mat', '-emg.mat')
    if not os.path.exists(dataset_base + emg_file):
        print('\nNo EMG file for ' + file)
        continue
    emg_samples= preprocessing_EMG(emg_file, emg_l_freq, emg_h_freq, emg_rsfreq, emg_samp_rate, seg_freq, window_time)

    kin_file = file.replace('-eeg.mat', '-kin.mat')
    if not os.path.exists(dataset_base + kin_file):
        print('\nNo KIN file for ' + file)
        continue
    label = preprocessing_KIN(kin_file, kin_rsfreq, seg_freq, window_time)
    
    eeg_samples = preprocessing_EEG(file, eeg_l_freq, eeg_h_freq, eeg_rsfreq, eeg_samp_rate, seg_freq, window_time)

    eog_samples = preprocessing_EOG(file, eog_l_freq, eog_h_freq, eog_rsfreq, eog_samp_rate, seg_freq, window_time)

    print(eeg_samples.shape[0], emg_samples.shape[0], eog_samples.shape[0], label.shape[0])
    if not (max(eeg_samples.shape[0], emg_samples.shape[0], eog_samples.shape[0], label.shape[0]) - min(eeg_samples.shape[0], emg_samples.shape[0], eog_samples.shape[0], label.shape[0])) < 2:
        continue
    emg_labels = [''] + [str(i) for i in range(2, 13)]
    save_path = dataset_base.replace('raw', 'preprocessed') + sbj + '-' + trial + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(eeg_samples.shape[0]):
        dump_path = save_path + sbj + '-' + trial + '_' + str(i) + '.pkl'
        if not os.path.exists(dump_path):
            with open(dump_path, 'wb') as f:
                pickle.dump(
                    {
                        "EEG": eeg_samples[i],
                        "EEG_ch_names": EEG_order,
                        "EMG": emg_samples[i],
                        "EMG_ch_names": ['EMG' + str(j) for j in emg_labels],
                        "EOG": eog_samples[i],
                        "EOG_ch_names": EOG_channels,
                        "Y": label[i],
                    },
                    f,
                )
