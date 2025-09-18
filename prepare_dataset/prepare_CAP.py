from pathlib import Path
import mne
import pickle
import os
from multiprocessing import Pool


EEG_channels = ['A1', 'F2-F4', 'T4', 'O2', 'F7-T3', 'F8', 'FP2-F4', 'F3-C3', 'T3', 'FP1', 'C3', 'Fp2', 'F7', 'C4', 'O2-A1', 'P4-O2', 'F1-F3', 'FP1-F3', 'P3-O1', 'F3', 'T6', 'A2', 'C4-A1', 'C4-P4', 'O1', 'P3', 'O1-A2', 'Fp2-F4', 'F4-C4', 'T3-T5', 'F4', 'T4-T6', 'T5', 'C3-P3', 'P4', 'F8-T4', 'C3-A2']
EOG_channels = ['ROC-LOC']
EMG_channels = ['EMG1-EMG2']
ECG_channels = ['ECG1-ECG2']

channels = [['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'Fp2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC', 'T3-T5', 'T4-T6'],
['C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F4-C4', 'Fp2-F4', 'P4-O2', 'ROC-LOC'],
['C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F4-C4', 'F7-T3', 'F8-T4', 'Fp2-F4', 'P4-O2', 'ROC-LOC'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'Fp2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F1-F3', 'F2-F4', 'F3-C3', 'F4-C4', 'P3-O1', 'P4-O2', 'ROC-LOC'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'FP1-F3', 'Fp2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'FP2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC', 'T3-T5', 'T4-T6'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'FP2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC', 'T3-T5', 'T4'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'Fp2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC', 'T4-T6'],
['C3-P3', 'C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F3-C3', 'F4-C4', 'FP1-F3', 'Fp2-F4', 'P3-O1', 'P4-O2', 'ROC-LOC', 'T3-T5', 'T4-T6'],
['C4-A1', 'C4-P4', 'ECG1-ECG2', 'EMG1-EMG2', 'F2-F4', 'F4-C4', 'P4-O2', 'ROC-LOC']]

all_configures = {}
cnt = 0

rawDataPath = Path('./CAP/physionet.org')
group = rawDataPath.rglob('*.edf')
dump_folder = './CAP/preprocessed'

# preprocessing parameters
eeg_l_freq = 0.1
eeg_h_freq = 75.0
eeg_rsfreq = 200

eog_l_freq = 0.1
eog_h_freq = 75.0
eog_rsfreq = 200

ecg_l_freq = 0.5
ecg_h_freq = 60.0
ecg_rsfreq = 500

emg_l_freq = 5
emg_h_freq = 200.0
emg_rsfreq = 500


def preprocessing_EEG(edfFilePath, sorted_ch_names, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in sorted_ch_names:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)
    raw.reorder_channels(sorted_ch_names)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    ch_names = [ch.upper() for ch in raw.ch_names]
    return eegData, ch_names


def preprocessing_EOG(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)

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
    return eogData, ['HEO']


def preprocessing_EMG(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EMG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=None, n_jobs=5)
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
    return emgData, ['EMG']


def preprocessing_ECG(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in ECG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    ecgData = raw.get_data(units='uV')
    return ecgData, ['ECG']


def preprocessing_edf(edfFilePath, sorted_ch_names):
    eegData, eegCh = preprocessing_EEG(edfFilePath, sorted_ch_names, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
    eogData, eogCh = preprocessing_EOG(edfFilePath, eog_l_freq, eog_h_freq, eog_rsfreq)
    ecgData, ecgCh = preprocessing_ECG(edfFilePath, ecg_l_freq, ecg_h_freq, ecg_rsfreq)
    emgData, emgCh = preprocessing_EMG(edfFilePath, emg_l_freq, emg_h_freq, emg_rsfreq)
    return eegData, eegCh, eogData, eogCh, ecgData, ecgCh, emgData, emgCh


def process(edfFile):
    global cnt, all_configures
    print(f'processing {edfFile.name}')
    raw = mne.io.read_raw_edf(edfFile, verbose=False)
    if EOG_channels[0] not in raw.ch_names or ECG_channels[0] not in raw.ch_names or EMG_channels[0] not in raw.ch_names:
        return
    drop_chs = []
    for ch in raw.ch_names:
        if ch not in EEG_channels + EOG_channels + EMG_channels + ECG_channels:
            drop_chs.append(ch)
    if drop_chs == raw.ch_names:
        return
    raw.drop_channels(drop_chs)
    sorted_ch_names = sorted(raw.ch_names)
    sorted_ch_names.remove('ROC-LOC')
    sorted_ch_names.remove('EMG1-EMG2')
    sorted_ch_names.remove('ECG1-ECG2')
    if str(sorted_ch_names) not in list(all_configures.keys()):
        all_configures[str(sorted_ch_names)] = str(cnt)
        cnt += 1
        if not os.path.exists(os.path.join(dump_folder, all_configures[str(sorted_ch_names)])):
            os.makedirs(os.path.join(dump_folder, all_configures[str(sorted_ch_names)]))
    folder = all_configures[str(sorted_ch_names)]
    
    eegData, eegCh, eogData, eogCh, ecgData, ecgCh, emgData, emgCh = preprocessing_edf(edfFile, sorted_ch_names)

    eegData = eegData[:, 10*eeg_rsfreq:-10*eeg_rsfreq]
    eogData = eogData[:, 10*eog_rsfreq:-10*eog_rsfreq]
    ecgData = ecgData[:, 10*ecg_rsfreq:-10*ecg_rsfreq]
    emgData = emgData[:, 10*emg_rsfreq:-10*emg_rsfreq]

    time = (512 // len(eegCh))
    eeg_time_length = time * eeg_rsfreq
    eog_time_length = time * eog_rsfreq
    ecg_time_length = time * ecg_rsfreq
    emg_time_length = time * emg_rsfreq
    for i in range(eegData.shape[1] // eeg_time_length):
        dump_path = os.path.join(
            dump_folder, folder, edfFile.name.split('.')[0] + "_" + str(i) + ".pkl"
        )
        pickle.dump(
            {
                "EEG": eegData[:, i * eeg_time_length : (i + 1) * eeg_time_length],
                "EEG_ch_names": eegCh,
                "EOG": eogData[:, i * eog_time_length : (i + 1) * eog_time_length],
                "EOG_ch_names": eogCh,
                "ECG": ecgData[:, i * ecg_time_length : (i + 1) * ecg_time_length],
                "ECG_ch_names": ecgCh,
                "EMG": emgData[:, i * emg_time_length : (i + 1) * emg_time_length],
                "EMG_ch_names": emgCh,
            },
            open(dump_path, "wb"),
        )

group = [g for g in group]
# split and dump in parallel
for g in group:
    process(g)

print(all_configures)