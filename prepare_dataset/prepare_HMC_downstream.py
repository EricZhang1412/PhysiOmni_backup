import mne
import numpy as np
import os
import pickle
import pandas as pd


EEG_channels = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
EOG_channels = ['EOG E1-M2', 'EOG E2-M2']
ECG_channels = ['ECG']
EMG_channels = ['EMG chin']

symbols_hmc = {' Sleep stage W': 0, ' Sleep stage N1': 1, ' Sleep stage N2': 2, ' Sleep stage N3': 3,' Sleep stage R': 4}


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


def BuildEvents(EEG_signals, EOG_signals, ECG_signals, EMG_signals, EEG_times, EOG_times, ECG_times, EMG_times, EventData):
    [numEvents, z] = EventData.shape 
    fs = 200.0

    EEG_features = np.zeros([numEvents - 2, EEG_signals.shape[0], int(eeg_rsfreq) * 30])
    EOG_features = np.zeros([numEvents - 2, EOG_signals.shape[0], int(eog_rsfreq) * 30])
    ECG_features = np.zeros([numEvents - 2, ECG_signals.shape[0], int(ecg_rsfreq) * 30])
    EMG_features = np.zeros([numEvents - 2, EMG_signals.shape[0], int(emg_rsfreq) * 30])
    labels = np.zeros([numEvents - 2, 1])
    i = 0
    for _, row in EventData.iterrows():
        if row[' Duration'] != 30:
            continue
        start = np.where((EEG_times) >= row[' Recording onset'])[0][0]
        end = np.where((EEG_times) >= (row[' Recording onset'] + row[' Duration']))[0][0]
        EEG_features[i, :] = EEG_signals[:, start:end]
        start = np.where((EOG_times) >= row[' Recording onset'])[0][0]
        end = np.where((EOG_times) >= (row[' Recording onset'] + row[' Duration']))[0][0]
        EOG_features[i, :] = EOG_signals[:, start:end]
        start = np.where((ECG_times) >= row[' Recording onset'])[0][0]
        end = np.where((ECG_times) >= (row[' Recording onset'] + row[' Duration']))[0][0]
        ECG_features[i, :] = ECG_signals[:, start:end]
        start = np.where((EMG_times) >= row[' Recording onset'])[0][0]
        end = np.where((EMG_times) >= (row[' Recording onset'] + row[' Duration']))[0][0]
        EMG_features[i, :] = EMG_signals[:, start:end]

        labels[i, :] = symbols_hmc[row[' Annotation']]
        i += 1
    return [EEG_features, EOG_features, ECG_features, EMG_features, labels]


def readEDF(fileName, used_channels, l_freq, h_freq, rsfreq):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    useless_chs = []
    for ch in Rawdata.ch_names:
        if ch not in used_channels:
            useless_chs.append(ch)
    Rawdata.drop_channels(useless_chs)
    if used_channels is not None and len(used_channels) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(used_channels)
    if Rawdata.ch_names != used_channels:
        raise ValueError

    Rawdata.filter(l_freq=l_freq, h_freq=h_freq)
    Rawdata.notch_filter(50.0)
    if h_freq >= 100:
        try:
            Rawdata = Rawdata.notch_filter(100.0, n_jobs=5)
            Rawdata = Rawdata.notch_filter(150.0, n_jobs=5)
            Rawdata = Rawdata.notch_filter(200.0, n_jobs=5)
        except:
            pass
    Rawdata.resample(rsfreq, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    Rawdata.close()
    return [signals, times]


def load_up_objects(fileList, Features, Labels, OutDir):
    for fname in fileList:
        print("\t%s" % fname)
        try:
            [EEG_signals, EEG_times] = readEDF(fname, EEG_channels, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
            [EOG_signals, EOG_times] = readEDF(fname, EOG_channels, eog_l_freq, eog_h_freq, eog_rsfreq)
            EOG_signals = EOG_signals[[0], :] - EOG_signals[[1], :]
            [ECG_signals, ECG_times] = readEDF(fname, ECG_channels, ecg_l_freq, ecg_h_freq, ecg_rsfreq)
            [EMG_signals, EMG_times] = readEDF(fname, EMG_channels, emg_l_freq, emg_h_freq, emg_rsfreq)

            labelFile = fname[0:-4] + "_sleepscoring.txt"
            labelData = pd.read_csv(labelFile)
        except (ValueError, KeyError):
            print("something funky happened in " + fname)
            continue
        EEG_signals, EOG_signals, ECG_signals, EMG_signals, labels = BuildEvents(EEG_signals, EOG_signals, ECG_signals, EMG_signals, \
                                                                                 EEG_times, EOG_times, ECG_times, EMG_times, labelData)

        for idx, (EEG_signal, EOG_signal, ECG_signal, EMG_signal, label) in enumerate(
            zip(EEG_signals, EOG_signals, ECG_signals, EMG_signals, labels)
        ):
            sample = {
                "EEG": EEG_signal,
                "EEG_ch_names": [name.split(' ')[-1].split('-')[0] for name in EEG_channels],
                "EOG": EOG_signal,
                "EOG_ch_names": ['HEO'],
                "ECG": ECG_signal,
                "ECG_ch_names": ['ECG'],
                "EMG": EMG_signal,
                "EMG_ch_names": ['EMG'],
                "Y": int(label),
            }
            #print(EEG_signal.shape, EOG_signal.shape, ECG_signal.shape, EMG_signal.shape)

            save_pickle(
                sample,
                os.path.join(
                    OutDir, fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                ),
            )

    return Features, Labels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


root = "./HMC/physionet.org/files/hmc-sleep-staging/1.1/recordings"
out_dir = './HMC/preprocessed'
train_out_dir = os.path.join(out_dir, "train")
eval_out_dir = os.path.join(out_dir, "eval")
test_out_dir = os.path.join(out_dir, "test")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)
if not os.path.exists(test_out_dir):
    os.makedirs(test_out_dir)

edf_files = []
for dirName, subdirList, fileList in os.walk(root):
    for fname in fileList:
        if len(fname) == 9 and fname[-4:] == ".edf":
            edf_files.append(os.path.join(dirName, fname))
edf_files.sort()

train_files = edf_files[:100]
eval_files = edf_files[100:125]
test_files = edf_files[125:]

fs = 200
TrainFeatures = np.empty(
    (0, 4, fs * 30)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
load_up_objects(
    train_files, TrainFeatures, TrainLabels, train_out_dir
)

fs = 200
EvalFeatures = np.empty(
    (0, 4, fs * 30)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
load_up_objects(
    eval_files, EvalFeatures, EvalLabels, eval_out_dir
)

fs = 200
TestFeatures = np.empty(
    (0, 4, fs * 30)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TestLabels = np.empty([0, 1])
load_up_objects(
    test_files, TestFeatures, TestLabels, test_out_dir
)
