from pathlib import Path
import mne
import pickle
import os
from tqdm import tqdm
from multiprocessing import Pool


channels = [['EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG LOC-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EMG-REF'], 
            ['EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG LOC-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EMG-REF'], 
            ['ECG EKG-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EMG-REF'], 
            ['ECG EKG-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG LOC-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF', 'EEG ROC-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF']]

dump_folder = Path('./TUH')
rawDataPath = Path('./tuh_eeg')
group = rawDataPath.rglob('*.edf')

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

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG EKG1-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'PULSE RATE', \
                    'EEG 1X10_LAT_01', '1X10_LAT_02', '1X10_LAT_03', '1X10_LAT_04', '1X10_LAT_05', 'X1', 'PG1', 'PG2', 'RESP THORAX-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
EOG_channels = ['EEG LOC-REF', 'EEG ROC-REF']
ECG_channels = ['ECG EKG-REF']
EMG_channels = ['EMG-REF']

all_configures = {}
cnt = 0
duration = 0

# channel number * rsfreq

def preprocessing_EEG(edfFilePath, sorted_ch_names, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    # reading edf
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
    raw = raw.notch_filter(60.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names


def preprocessing_EOG(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)

    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EOG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)
    raw.reorder_channels(EOG_channels)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(60.0, n_jobs=5)
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
    try:
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    except:
        raw = raw.filter(l_freq=l_freq, h_freq=None, n_jobs=5)
    raw = raw.notch_filter(60.0, n_jobs=5)
    try:
        raw = raw.notch_filter(120.0, n_jobs=5)
        raw = raw.notch_filter(180.0, n_jobs=5)
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
    #raw = raw.notch_filter(60.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    ecgData = raw.get_data(units='uV')
    return ecgData, ['ECG']


def preprocessing_edf(edfFilePath, sorted_ch_names, contain_EOG, contain_ECG, contain_EMG):
    eegData, eegCh = preprocessing_EEG(edfFilePath, sorted_ch_names, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
    eogData, eogCh, ecgData, ecgCh, emgData, emgCh = None, None, None, None, None, None
    if contain_EOG:
        eogData, eogCh = preprocessing_EOG(edfFilePath, eog_l_freq, eog_h_freq, eog_rsfreq)
        eogData = eogData[[1], :] - eogData[[0], :]
    if contain_ECG:
        ecgData, ecgCh = preprocessing_ECG(edfFilePath, ecg_l_freq, ecg_h_freq, ecg_rsfreq)
    if contain_EMG:
        emgData, emgCh = preprocessing_EMG(edfFilePath, emg_l_freq, emg_h_freq, emg_rsfreq)
    return eegData, eegCh, eogData, eogCh, ecgData, ecgCh, emgData, emgCh


def process(edfFile):
    print(f'processing {edfFile.name}')
    global cnt, all_configures, duration
    # reading edf
    raw = mne.io.read_raw_edf(edfFile, preload=False)
    if raw.ch_names[0].split('-')[-1] == 'LE':
        return None, raw.ch_names
    if drop_channels is not None:
        useless_chs = []
        for ch in raw.ch_names:
            if ch in drop_channels:
                useless_chs.append(ch)
        raw.drop_channels(useless_chs)
    sorted_ch_names = sorted(raw.ch_names)
    num_modality = 1
    contain_EMG, contain_ECG, contain_EOG = False, False, False
    if 'EMG-REF' in sorted_ch_names:
        num_modality += 1
        contain_EMG = True
    if 'ECG EKG-REF' in sorted_ch_names:
        num_modality += 1
        contain_ECG = True
    if 'EEG ROC-REF' in sorted_ch_names and 'EEG LOC-REF' in sorted_ch_names:
        num_modality += 1
        contain_EOG = True
    if num_modality < 3:
        return
    duration += raw.n_times / raw.info['sfreq'] / 3600
    if str(sorted_ch_names) not in list(all_configures.keys()):
        all_configures[str(sorted_ch_names)] = str(cnt)
        cnt += 1
        if not os.path.exists(os.path.join(dump_folder, all_configures[str(sorted_ch_names)])):
            os.makedirs(os.path.join(dump_folder, all_configures[str(sorted_ch_names)]))
    folder = all_configures[str(sorted_ch_names)]
    if 'EMG-REF' in sorted_ch_names:
        sorted_ch_names.remove('EMG-REF')
    if 'ECG EKG-REF' in sorted_ch_names:
        sorted_ch_names.remove('ECG EKG-REF')
    if 'EEG ROC-REF' in sorted_ch_names and 'EEG LOC-REF' in sorted_ch_names:
        sorted_ch_names.remove('EEG ROC-REF')
        sorted_ch_names.remove('EEG LOC-REF')

    eegData, eegCh, eogData, eogCh, ecgData, ecgCh, emgData, emgCh = preprocessing_edf(edfFile, sorted_ch_names, contain_EOG, contain_ECG, contain_EMG)

    eegCh = [s.split(' ')[-1].split('-')[0] for s in eegCh]

    eegData = eegData[:, :-10*eeg_rsfreq]
    if eogData is not None:
        eogData = eogData[:, :-10*eog_rsfreq]
    if ecgData is not None:
        ecgData = ecgData[:, :-10*ecg_rsfreq]
    if emgData is not None:
        emgData = emgData[:, :-10*emg_rsfreq]

    time = (512 // len(eegCh))
    eeg_time_length = time * eeg_rsfreq
    eog_time_length = time * eog_rsfreq
    ecg_time_length = time * ecg_rsfreq
    emg_time_length = time * emg_rsfreq
    for i in range(eegData.shape[1] // eeg_time_length):
        dump_path = os.path.join(
            dump_folder, folder, edfFile.name.split('.')[0] + "_" + str(i) + ".pkl"
        )
        save = {
            "EEG": eegData[:, i * eeg_time_length : (i + 1) * eeg_time_length],
            "EEG_ch_names": eegCh,
        }
        if eogData is not None:
            save.update({
                "EOG": eogData[:, i * eog_time_length : (i + 1) * eog_time_length],
                "EOG_ch_names": ['HEO'],
            })
        if ecgData is not None:
            save.update({
                "ECG": ecgData[:, i * ecg_time_length : (i + 1) * ecg_time_length],
                "ECG_ch_names": ['ECG'],
            })
        if emgData is not None:
            save.update({
                "EMG": emgData[:, i * emg_time_length : (i + 1) * emg_time_length],
                "EMG_ch_names": ['EMG'],
            })
        pickle.dump(
            save,
            open(dump_path, "wb"),
        )

group = [g for g in group]
# split and dump in parallel
for g in group:
    process(g)
print(all_configures)
print(duration)
