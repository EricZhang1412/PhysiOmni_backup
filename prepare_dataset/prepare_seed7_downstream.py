import mne
import os
import csv
import datetime
import pickle


def get_montage(montage_file_path: str = 'channel_62_pos.locs') -> mne.channels.DigMontage:
    """
    Return the montage specified by the param montage_file_path
    """
    montage = mne.channels.read_custom_montage(montage_file_path)
    return montage


EEG_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', \
                'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', \
                    'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', \
                        'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', \
                            'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
EOG_channels = ['VEO', 'HEO']
ECG_channels = ['ECG']

eeg_l_freq = 0.1
eeg_h_freq = 75.0
eeg_rsfreq = 200

eog_l_freq = 0.1
eog_h_freq = 75.0
eog_rsfreq = 200

ecg_l_freq = 0.5
ecg_h_freq = 60.0
ecg_rsfreq = 500

video_order = ['happy', 'neutral', 'disgust', 'sad', 'anger', 'anger', 'sad', 'disgust', 'neutral', 'happy'] * 2 +\
    ['anger', 'sad', 'fear', 'neutral', 'surprise', 'surprise', 'neutral', 'fear', 'sad', 'anger'] * 2 +\
    ['happy', 'surprise', 'disgust', 'fear', 'anger', 'anger', 'fear', 'disgust', 'surprise', 'happy'] * 2 +\
    ['disgust', 'sad', 'fear', 'surprise', 'happy', 'happy', 'surprise', 'fear', 'sad', 'disgust'] * 2
label_dict = {'happy': 0, 'surprise': 1, 'neutral': 2, 'sad': 3, 'disgust': 4, 'fear': 5, 'anger': 6}
label = [label_dict[i] for i in video_order]


def preprocessing_EEG(cntFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_cnt(cntFilePath, preload=False)
    raw_file = cntFilePath.split('/')[-1]
    useless_chs = []
    for ch in raw.ch_names:
        if ch not in EEG_channels:
            useless_chs.append(ch)

    raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)

    # Get triggers for each trial
    trigger, _ = mne.events_from_annotations(raw)

    eegData, times = raw.get_data(units='uV', return_times=True)

    t = trigger[:, 0]

    # The triggers in cnt file are not accurate for these two files, so we use trigger files instead
    if raw_file == "14_20221015_1.cnt":
        t = []
        start = datetime.datetime.strptime('14:25:34', '%H:%M:%S')
        with open('./SEED-VII/save_info/14_20221015_1_trigger_info.csv') as f:
            trigger = csv.reader(f)
            for row in trigger:
                end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                time_diff = end.timestamp() - start.timestamp()
                t.append(int(round(time_diff * sfreq)))
    elif raw_file == "9_20221111_3.cnt":
        t = []
        start = datetime.datetime.strptime('14:01:27', '%H:%M:%S')
        with open('./SEED-VII/save_info/9_20221111_3_trigger_info.csv') as f:
            trigger = csv.reader(f)
            for row in trigger:
                end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                time_diff = end.timestamp() - start.timestamp()
                t.append(int(round(time_diff * sfreq)))
    return eegData, raw.ch_names, t


def preprocessing_EOG(cntFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_cnt(cntFilePath, preload=False)
    raw_file = cntFilePath.split('/')[-1]
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
    
    # Get triggers for each trial
    trigger, _ = mne.events_from_annotations(raw)

    eogData, times = raw.get_data(units='uV', return_times=True)

    t = trigger[:, 0]

    # The triggers in cnt file are not accurate for these two files, so we use trigger files instead
    if raw_file == "14_20221015_1.cnt":
        t = []
        start = datetime.datetime.strptime('14:25:34', '%H:%M:%S')
        with open('./SEED-VII/save_info/14_20221015_1_trigger_info.csv') as f:
            trigger = csv.reader(f)
            for row in trigger:
                end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                time_diff = end.timestamp() - start.timestamp()
                t.append(int(round(time_diff * sfreq)))
    elif raw_file == "9_20221111_3.cnt":
        t = []
        start = datetime.datetime.strptime('14:01:27', '%H:%M:%S')
        with open('./SEED-VII/save_info/9_20221111_3_trigger_info.csv') as f:
            trigger = csv.reader(f)
            for row in trigger:
                end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                time_diff = end.timestamp() - start.timestamp()
                t.append(int(round(time_diff * sfreq)))
    return eogData, raw.ch_names, t


def preprocessing_ECG(cntFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    raw = mne.io.read_raw_cnt(cntFilePath, preload=False)
    raw_file = cntFilePath.split('/')[-1]
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
    
    # Get triggers for each trial
    trigger, _ = mne.events_from_annotations(raw)

    ecgData, times = raw.get_data(units='uV', return_times=True)

    t = trigger[:, 0]

    # The triggers in cnt file are not accurate for these two files, so we use trigger files instead
    if raw_file == "14_20221015_1.cnt":
        t = []
        start = datetime.datetime.strptime('14:25:34', '%H:%M:%S')
        with open('./SEED-VII/save_info/14_20221015_1_trigger_info.csv') as f:
            trigger = csv.reader(f)
            for row in trigger:
                end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                time_diff = end.timestamp() - start.timestamp()
                t.append(int(round(time_diff * sfreq)))
    elif raw_file == "9_20221111_3.cnt":
        t = []
        start = datetime.datetime.strptime('14:01:27', '%H:%M:%S')
        with open('./SEED-VII/save_info/9_20221111_3_trigger_info.csv') as f:
            trigger = csv.reader(f)
            for row in trigger:
                end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                time_diff = end.timestamp() - start.timestamp()
                t.append(int(round(time_diff * sfreq)))
    return ecgData, raw.ch_names, t


def preprocessing_cnt(cntFilePath):
    eegData, eegCh, eegt = preprocessing_EEG(cntFilePath, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
    eogData, eogCh, eogt = preprocessing_EOG(cntFilePath, eog_l_freq, eog_h_freq, eog_rsfreq)
    ecgData, ecgCh, ecgt = preprocessing_ECG(cntFilePath, ecg_l_freq, ecg_h_freq, ecg_rsfreq)
    return eegData, eegCh, eegt, eogData, eogCh, eogt, ecgData, ecgCh, ecgt


if __name__ == '__main__':
    data_path = './SEED-VII/EEG_raw'
    raw_files = os.listdir(data_path)

    save_path = './SEED-VII/SEED-VII-downstream'
    train_save_path = os.path.join(save_path, 'train')
    val_save_path = os.path.join(save_path, 'val')
    test_save_path = os.path.join(save_path, 'test')
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    file_dict = {}
    for file in raw_files:
        if file[:-15] not in file_dict.keys():
            file_dict[file[:-15]] = [file]
        else:
            file_dict[file[:-15]].append(file)
    #print(file_dict)

    for key, value in file_dict.items():
        EEG_preprocessed = {}
        EEG_features = {}
        for raw_file in value:
            raw_path = os.path.join(data_path, raw_file)
            raw = mne.io.read_raw_cnt(raw_path, eog=['HEO', 'VEO'], ecg=['ECG'])

            print(f'processing {raw_file}')
            eegData, eegCh, eegt, eogData, eogCh, eogt, ecgData, ecgCh, ecgt = preprocessing_cnt(raw_path)

            session_idx = int(raw_file[-5]) - 1
            for i in range(20):
                print(f'Subject {key}, video index {session_idx * 20 + i + 1}')
                EEG_clip = eegData[:, eegt[2 * i]:eegt[2 * i + 1]]
                EOG_clip = eogData[:, eogt[2 * i]:eogt[2 * i + 1]]
                ECG_clip = ecgData[:, ecgt[2 * i]:ecgt[2 * i + 1]]

                time = 1
                eeg_time_length = time * eeg_rsfreq
                eog_time_length = time * eog_rsfreq
                ecg_time_length = time * ecg_rsfreq
                for j in range(EEG_clip.shape[1] // eeg_time_length):
                    EEG_sample = EEG_clip[:, j * eeg_time_length:(j + 1) * eeg_time_length]
                    EOG_sample = EOG_clip[:, j * eog_time_length:(j + 1) * eog_time_length]
                    ECG_sample = ECG_clip[:, j * ecg_time_length:(j + 1) * ecg_time_length]

                    if i + 1 <= 10:
                        dump_folder = train_save_path
                    elif i + 1 <= 15:
                        dump_folder = val_save_path
                    else:
                        dump_folder = test_save_path
                    dump_path = os.path.join(
                        dump_folder, raw_file.split('.')[0] + "_" + str(i) + "_" + str(j) + ".pkl"
                    )
                    pickle.dump(
                        {
                            "EEG": EEG_sample,
                            "EEG_ch_names": eegCh,
                            "EOG": EOG_sample,
                            "EOG_ch_names": eogCh,
                            "ECG": ECG_sample,
                            "ECG_ch_names": ecgCh,
                            "Y": label[session_idx * 20 + i],
                        },
                        open(dump_path, "wb"),
                    )
