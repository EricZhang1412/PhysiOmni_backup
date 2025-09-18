from torch.utils.data import Dataset
import torch
from einops import rearrange
import pickle
import os


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
    'T1', 'T2', 'I1', 'I2', 'HEO', 'VEO', 'ECG', 'EMG', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', \
    'EMG7', 'EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12', \
    'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'P3-O1', 'P4-O2', 'T3-T5', 'T4-T6', \
    'C3-P3', 'C4-A1', 'C4-P4', 'F2-F4', 'FPZ-CZ', 'PZ-OZ', 'FP2-F4', 'F1-F3'
]


def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans


class PickleLoader(Dataset):
    def __init__(self, root, files, contain_EEG=True, contain_EOG=False, contain_ECG=False, contain_EMG=False, VQ_training=False):
        self.root = root
        self.files = files
        self.contain_EEG = contain_EEG
        self.contain_EOG = contain_EOG
        self.contain_ECG = contain_ECG
        self.contain_EMG = contain_EMG
        self.VQ_training = VQ_training

    def __len__(self):
        return len(self.files)
    
    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def get_chans(self, ch_names):
            chans = []
            for ch_name in ch_names:
                chans.append(standard_1020.index(ch_name))
            return chans

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))

        data = {}

        if self.contain_EEG:
            EEG = sample["EEG"]
            EEG_ch_names = sample["EEG_ch_names"]
            EEG = torch.FloatTensor(EEG / 100)

            EEG_time = EEG.size(1) // 200
            EEG_input_time = [i for _ in range(EEG.size(0)) for i in range(EEG_time)]
            EEG_X = rearrange(EEG, 'N (A T) -> (N A) T', T=200)

            EEG_input_chans = [i for i in list(EEG_ch_names) for _ in range(EEG_time)]
            EEG_input_chans = torch.IntTensor(self.get_chans(EEG_input_chans))
            EEG_input_time = torch.IntTensor(EEG_input_time)

            data['EEG_X'] = EEG_X
            data['EEG_input_chans'] = EEG_input_chans
            data['EEG_input_time'] = EEG_input_time
            if self.VQ_training:
                #data['EEG_Y'] = self.std_norm(EEG_X)
                x_fft = torch.fft.fft(EEG_X, dim=-1)
                amplitude = torch.abs(x_fft)
                data['EEG_Y'] = self.std_norm(amplitude)

        if self.contain_EOG:
            EOG = sample["EOG"]
            EOG_ch_names = sample["EOG_ch_names"]
            EOG = torch.FloatTensor(EOG / 100)

            EOG_time = EOG.size(1) // 100
            EOG_input_time = [i for _ in range(EOG.size(0)) for i in range(EOG_time)]
            EOG_X = rearrange(EOG, 'N (A T) -> (N A) T', T=100)

            EOG_input_chans = [i for i in list(EOG_ch_names) for _ in range(EOG_time)]
            EOG_input_chans = torch.IntTensor(self.get_chans(EOG_input_chans))
            EOG_input_time = torch.IntTensor(EOG_input_time)

            data['EOG_X'] = EOG_X
            data['EOG_input_chans'] = EOG_input_chans
            data['EOG_input_time'] = EOG_input_time
            if self.VQ_training:
                data['EOG_Y'] = self.std_norm(EOG_X)

        if self.contain_ECG:
            ECG = sample["ECG"]
            ECG_ch_names = sample["ECG_ch_names"]
            ECG = torch.FloatTensor(ECG / 100)
            
            ECG_time = ECG.size(1) // 100
            ECG_input_time = [i for _ in range(ECG.size(0)) for i in range(ECG_time)]
            ECG_X = rearrange(ECG, 'N (A T) -> (N A) T', T=100)

            ECG_input_chans = [i for i in list(ECG_ch_names) for _ in range(ECG_time)]
            ECG_input_chans = torch.IntTensor(self.get_chans(ECG_input_chans))
            ECG_input_time = torch.IntTensor(ECG_input_time)

            data['ECG_X'] = ECG_X
            data['ECG_input_chans'] = ECG_input_chans
            data['ECG_input_time'] = ECG_input_time
            if self.VQ_training:
                data['ECG_Y'] = self.std_norm(ECG_X)

        if self.contain_EMG:
            EMG = sample["EMG"]
            EMG_ch_names = sample["EMG_ch_names"]
            EMG = torch.FloatTensor(EMG / 100)

            EMG_time = EMG.size(1) // 100
            EMG_input_time = [i for _ in range(EMG.size(0)) for i in range(EMG_time)]
            EMG_X = rearrange(EMG, 'N (A T) -> (N A) T', T=100)

            EMG_input_chans = [i for i in list(EMG_ch_names) for _ in range(EMG_time)]
            EMG_input_chans = torch.IntTensor(self.get_chans(EMG_input_chans))
            EMG_input_time = torch.IntTensor(EMG_input_time)

            data['EMG_X'] = EMG_X
            data['EMG_input_chans'] = EMG_input_chans
            data['EMG_input_time'] = EMG_input_time
            if self.VQ_training:
                #data['EMG_Y'] = self.std_norm(EMG_X)
                x_fft = torch.fft.fft(EMG_X, dim=-1)
                amplitude = torch.abs(x_fft)
                data['EMG_Y'] = self.std_norm(amplitude)
        
        return data


class DownstreamLoader(Dataset):
    def __init__(self, root, files, name, contain_EEG=True, contain_EOG=False, contain_ECG=False, contain_EMG=False):
        self.root = root
        self.files = files
        self.name = name
        self.contain_EEG = contain_EEG
        self.contain_EOG = contain_EOG
        self.contain_ECG = contain_ECG
        self.contain_EMG = contain_EMG

    def __len__(self):
        return len(self.files)

    def get_chans(self, ch_names):
            chans = []
            for ch_name in ch_names:
                chans.append(standard_1020.index(ch_name))
            return chans

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))

        data = {}
        if self.name == 'FBM':
            data['Y'] = torch.FloatTensor(sample['Y'] / 90)
        else:
            data['Y'] = sample['Y']

        if self.contain_EEG:
            EEG = sample["EEG"]
            EEG_ch_names = sample["EEG_ch_names"]
            EEG = torch.FloatTensor(EEG / 100)

            EEG_time = EEG.size(1) // 200
            EEG_input_time = [i for _ in range(EEG.size(0)) for i in range(EEG_time)]
            EEG_X = rearrange(EEG, 'N (A T) -> (N A) T', T=200)

            EEG_input_chans = [i for i in list(EEG_ch_names) for _ in range(EEG_time)]
            EEG_input_chans = torch.IntTensor(self.get_chans(EEG_input_chans))
            EEG_input_time = torch.IntTensor(EEG_input_time)

            data['EEG_X'] = EEG_X
            data['EEG_input_chans'] = EEG_input_chans
            data['EEG_input_time'] = EEG_input_time

        if self.contain_EOG:
            EOG = sample["EOG"]
            EOG_ch_names = sample["EOG_ch_names"]
            EOG = torch.FloatTensor(EOG / 100)

            EOG_time = EOG.size(1) // 100
            EOG_input_time = [i for _ in range(EOG.size(0)) for i in range(EOG_time)]
            EOG_X = rearrange(EOG, 'N (A T) -> (N A) T', T=100)

            EOG_input_chans = [i for i in list(EOG_ch_names) for _ in range(EOG_time)]
            EOG_input_chans = torch.IntTensor(self.get_chans(EOG_input_chans))
            EOG_input_time = torch.IntTensor(EOG_input_time)

            data['EOG_X'] = EOG_X
            data['EOG_input_chans'] = EOG_input_chans
            data['EOG_input_time'] = EOG_input_time

        if self.contain_ECG:
            ECG = sample["ECG"]
            ECG_ch_names = sample["ECG_ch_names"]
            ECG = torch.FloatTensor(ECG / 100)
            
            ECG_time = ECG.size(1) // 100
            ECG_input_time = [i for _ in range(ECG.size(0)) for i in range(ECG_time)]
            ECG_X = rearrange(ECG, 'N (A T) -> (N A) T', T=100)

            ECG_input_chans = [i for i in list(ECG_ch_names) for _ in range(ECG_time)]
            ECG_input_chans = torch.IntTensor(self.get_chans(ECG_input_chans))
            ECG_input_time = torch.IntTensor(ECG_input_time)

            data['ECG_X'] = ECG_X
            data['ECG_input_chans'] = ECG_input_chans
            data['ECG_input_time'] = ECG_input_time

        if self.contain_EMG:
            EMG = sample["EMG"]
            EMG_ch_names = sample["EMG_ch_names"]
            EMG = torch.FloatTensor(EMG / 100)

            EMG_time = EMG.size(1) // 100
            EMG_input_time = [i for _ in range(EMG.size(0)) for i in range(EMG_time)]
            EMG_X = rearrange(EMG, 'N (A T) -> (N A) T', T=100)

            EMG_input_chans = [i for i in list(EMG_ch_names) for _ in range(EMG_time)]
            EMG_input_chans = torch.IntTensor(self.get_chans(EMG_input_chans))
            EMG_input_time = torch.IntTensor(EMG_input_time)

            data['EMG_X'] = EMG_X
            data['EMG_input_chans'] = EMG_input_chans
            data['EMG_input_time'] = EMG_input_time
        
        return data
