from pathlib import Path
import mne
import pickle
import os
from multiprocessing import Pool
import numpy as np


EEG_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
EOG_channels = ['EOG horizontal']
EMG_channels = ['EMG submental']

rawDataPath = Path('./Sleep-EDF')
group = rawDataPath.rglob('*PSG.edf')
dump_folder = './Sleep-EDF/preprocessed'


# preprocessing parameters
eeg_l_freq = 0.1
eeg_h_freq = 75.0
eeg_rsfreq = 200

eog_l_freq = 0.1
eog_h_freq = 75.0
eog_rsfreq = 200

emg_l_freq = 5
emg_h_freq = 200.0
emg_rsfreq = 500


def preprocessing_EEG(cntFilePath, tmin, tmax, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(cntFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EEG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=None, n_jobs=5)
    #raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    raw = raw.crop(tmin=tmin, tmax=tmax)
    eegData = raw.get_data(units='uV')
    return eegData, raw.ch_names


def preprocessing_EOG(cntFilePath, tmin, tmax, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(cntFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EOG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=None, n_jobs=5)
    #raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    raw = raw.crop(tmin=tmin, tmax=tmax)
    eogData = raw.get_data(units='uV')
    return eogData, raw.ch_names


def preprocessing_EMG(cntFilePath, tmin, tmax, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(cntFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EMG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=None, n_jobs=5)
    #raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    raw = raw.crop(tmin=tmin, tmax=tmax)
    emgData = raw.get_data(units='uV')
    return emgData, raw.ch_names


def preprocessing_edf(cntFilePath):
    raw = mne.io.read_raw_edf(cntFilePath, preload=False)
    useless_chs = []
    for ch in raw.ch_names:
        if ch not in (EMG_channels + EOG_channels + EEG_channels):
            useless_chs.append(ch)
    raw.drop_channels(useless_chs)
    raw.load_data()
    data = raw.get_data()
    std_dev = np.std(data, axis=0)
    empty_samples = np.where(std_dev != 0)[0]
    tmin = empty_samples[0] / raw.info['sfreq']
    tmax = empty_samples[-1] / raw.info['sfreq']

    eegData, eegCh = preprocessing_EEG(cntFilePath, tmin, tmax, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
    eogData, eogCh = preprocessing_EOG(cntFilePath, tmin, tmax, eog_l_freq, eog_h_freq, eog_rsfreq)
    emgData, emgCh = preprocessing_EMG(cntFilePath, tmin, tmax, emg_l_freq, emg_h_freq, emg_rsfreq)
    eegCh = [ch.upper().split(' ')[-1] for ch in eegCh]
    return eegData, eegCh, eogData, eogCh, emgData, emgCh


def process(cntFile):
    print(f'processing {cntFile.name}')
    if 'ST' not in cntFile.name:
        return
    eegData, eegCh, eogData, eogCh, emgData, emgCh = preprocessing_edf(cntFile)

    time = (204 // len(eegCh))
    eeg_time_length = time * eeg_rsfreq
    eog_time_length = time * eog_rsfreq
    emg_time_length = time * emg_rsfreq
    for i in range(eegData.shape[1] // eeg_time_length):
        dump_path = os.path.join(
            dump_folder, cntFile.name.split('.')[0] + "_" + str(i) + ".pkl"
        )
        pickle.dump(
            {
                "EEG": eegData[:, i * eeg_time_length : (i + 1) * eeg_time_length],
                "EEG_ch_names": eegCh,
                "EOG": eogData[:, i * eog_time_length : (i + 1) * eog_time_length],
                "EOG_ch_names": ['HEO'],
                "EMG": emgData[:, i * emg_time_length : (i + 1) * emg_time_length],
                "EMG_ch_names": ['EMG'],
            },
            open(dump_path, "wb"),
        )


group = [g for g in group]
# split and dump in parallel
with Pool(processes=12) as pool:
    # Use the pool.map function to apply the square function to each element in the numbers list
    pool.map(process, group)
