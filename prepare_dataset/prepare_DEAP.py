from pathlib import Path
import mne
import pickle
import os
from multiprocessing import Pool
import numpy as np


EEG_channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
EOG_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
EMG_channels = ['EXG5', 'EXG6', 'EXG7', 'EXG8']

rawDataPath = Path('./DEAP')
group = rawDataPath.rglob('*.bdf')
dump_folder = './DEAP/preprocessed'


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


def preprocessing_EEG(bdfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_bdf(bdfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EEG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)
    raw.reorder_channels(EEG_channels)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')
    return eegData, raw.ch_names


def preprocessing_EOG(bdfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_bdf(bdfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EOG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eogData = raw.get_data(units='uV')
    return eogData, raw.ch_names


def preprocessing_EMG(bdfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_bdf(bdfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EMG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    try:
        raw = raw.notch_filter(100.0, n_jobs=5)
        raw = raw.notch_filter(150.0, n_jobs=5)
        raw = raw.notch_filter(200.0, n_jobs=5)
    except:
        pass
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    emgData = raw.get_data(units='uV')
    return emgData, raw.ch_names


def preprocessing_bdf(bdfFilePath):
    eegData, eegCh = preprocessing_EEG(bdfFilePath, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
    eegCh = [ch.upper() for ch in eegCh]
    eogData, eogCh = preprocessing_EOG(bdfFilePath, eog_l_freq, eog_h_freq, eog_rsfreq)
    heo = eogData[1, :] - eogData[0, :]
    veo = eogData[2, :] - eogData[3, :]
    eogData = np.stack((heo, veo), axis=0)
    emgData, emgCh = preprocessing_EMG(bdfFilePath, emg_l_freq, emg_h_freq, emg_rsfreq)
    return eegData, eegCh, eogData, eogCh, emgData, emgCh


def process(bdfFile):
    print(f'processing {bdfFile.name}')
    eegData, eegCh, eogData, eogCh, emgData, emgCh = preprocessing_bdf(bdfFile)

    eegData = eegData[:, :-10*eeg_rsfreq]
    eogData = eogData[:, :-10*eog_rsfreq]
    emgData = emgData[:, :-10*emg_rsfreq]

    time = (512 // len(eegCh))
    eeg_time_length = time * eeg_rsfreq
    eog_time_length = time * eog_rsfreq
    emg_time_length = time * emg_rsfreq
    for i in range(eegData.shape[1] // eeg_time_length):
        dump_path = os.path.join(
            dump_folder, bdfFile.name.split('.')[0] + "_" + str(i) + ".pkl"
        )
        pickle.dump(
            {
                "EEG": eegData[:, i * eeg_time_length : (i + 1) * eeg_time_length],
                "EEG_ch_names": eegCh,
                "EOG": eogData[:, i * eog_time_length : (i + 1) * eog_time_length],
                "EOG_ch_names": ['HEO', 'VEO'],
                "EMG": emgData[:, i * emg_time_length : (i + 1) * emg_time_length],
                "EMG_ch_names": ['EMG', 'EMG2', 'EMG3', 'EMG4'],
            },
            open(dump_path, "wb"),
        )

group = [g for g in group]
# split and dump in parallel
with Pool(processes=10) as pool:
    # Use the pool.map function to apply the square function to each element in the numbers list
    pool.map(process, group)
