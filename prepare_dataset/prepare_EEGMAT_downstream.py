import mne
import numpy as np
import os
import pickle


ECG_channels = ['ECG ECG']
EEG_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']

eeg_l_freq = 0.1
eeg_h_freq = 75.0
eeg_rsfreq = 200

ecg_l_freq = 0.5
ecg_h_freq = 60.0
ecg_rsfreq = 500


def BuildEvents(EEG_signals, ECG_signals):
    [numChan, numPoints] = EEG_signals.shape
    numEvents = 29

    EEG_features = np.zeros([numEvents, numChan, int(eeg_rsfreq) * 4])
    for i in range(numEvents):
        start = i * 2 * eeg_rsfreq
        end = (i + 2) * 2 * eeg_rsfreq
        EEG_features[i, :] = EEG_signals[:, start:end]
    [numChan, numPoints] = ECG_signals.shape
    ECG_features = np.zeros([numEvents, numChan, int(ecg_rsfreq) * 4])
    for i in range(numEvents):
        start = i * 2 * ecg_rsfreq
        end = (i + 2) * 2 * ecg_rsfreq
        ECG_features[i, :] = ECG_signals[:, start:end]
    return EEG_features, ECG_features


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
    Rawdata.resample(rsfreq, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    Rawdata.close()
    return [signals, times]


def load_up_objects(fileList, Features, Labels, OutDir):
    for fname in fileList:
        print("\t%s" % fname)
        if fname[-5] == '1':
            label = 0
        elif fname[-5] == '2':
            label = 1
        try:
            [EEG_signals, EEG_times] = readEDF(fname, EEG_channels, eeg_l_freq, eeg_h_freq, eeg_rsfreq)
            [ECG_signals, ECG_times] = readEDF(fname, ECG_channels, ecg_l_freq, ecg_h_freq, ecg_rsfreq)
        except (ValueError, KeyError):
            print("something funky happened in " + fname)
            continue
        EEG_signals, ECG_signals = BuildEvents(EEG_signals, ECG_signals)

        for idx, (EEG_signal, ECG_signal) in enumerate(zip(EEG_signals, ECG_signals)):
            sample = {
                "EEG": EEG_signal,
                "EEG_ch_names": [name.upper().split(' ')[-1] for name in EEG_channels],
                "ECG": ECG_signal,
                "ECG_ch_names": ['ECG'],
                "Y": label,
            }
            print(EEG_signal.shape, ECG_signal.shape)

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


root = "./EEGMAT/physionet.org/files/eegmat/1.0.0"
out_dir = './EEGMAT/preprocessed'
train_out_dir = os.path.join(out_dir, "train")
eval_out_dir = os.path.join(out_dir, "val")
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
        if fname[-4:] == ".edf":
            edf_files.append(os.path.join(dirName, fname))
edf_files.sort()

train_files = edf_files[:52]
eval_files = edf_files[52:62]
test_files = edf_files[62:]

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
    (0, 19, fs * 4)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
load_up_objects(
    eval_files, EvalFeatures, EvalLabels, eval_out_dir
)

fs = 200
TestFeatures = np.empty(
    (0, 19, fs * 4)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TestLabels = np.empty([0, 1])
load_up_objects(
    test_files, TestFeatures, TestLabels, test_out_dir
)
