import math
import numpy as np
import os
import yaml
from dataset import PickleLoader, DownstreamLoader
#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from metrics import binary_metrics_fn, multiclass_metrics_fn
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import pearsonr


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def prepare_pretrain_dataset(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    datasets_list = []
    datasets = config['datasets']
    for dataset in datasets:
        contain_EEG = dataset['contain_EEG']
        contain_EOG = dataset['contain_EOG']
        contain_ECG = dataset['contain_ECG']
        contain_EMG = dataset['contain_EMG']
        path = dataset['path']
        files = os.listdir(os.path.join(path))
        datasets_list.append(PickleLoader(path, files, contain_EEG, contain_EOG, contain_ECG, contain_EMG, VQ_training=True))
    return datasets_list


def prepare_pretrain_VQ_dataset(root, contain_EEG=True, contain_EOG=False, contain_ECG=False, contain_EMG=False):
    files = os.listdir(os.path.join(root))
    train_files = files[:int(0.9 * len(files))]
    val_files = files[int(0.9 * len(files)):]

    print(len(train_files), len(val_files))

    train_dataset = PickleLoader(root, train_files, contain_EEG, contain_EOG, contain_ECG, contain_EMG, VQ_training=True)
    val_dataset = PickleLoader(root, val_files, contain_EEG, contain_EOG, contain_ECG, contain_EMG, VQ_training=True)

    return train_dataset, val_dataset


def prepare_SEED7_dataset(root, name, contain_EEG=True, contain_EOG=True, contain_ECG=True, contain_EMG=False):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = DownstreamLoader(os.path.join(root, "train"), train_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    test_dataset = DownstreamLoader(os.path.join(root, "test"), test_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    val_dataset = DownstreamLoader(os.path.join(root, "val"), val_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_HMC_dataset(root, name, contain_EEG=True, contain_EOG=True, contain_ECG=True, contain_EMG=True):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = DownstreamLoader(os.path.join(root, "train"), train_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    test_dataset = DownstreamLoader(os.path.join(root, "test"), test_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    val_dataset = DownstreamLoader(os.path.join(root, "eval"), val_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_FBM_dataset(root, name, contain_EEG=True, contain_EOG=True, contain_ECG=False, contain_EMG=False):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = DownstreamLoader(os.path.join(root, "train"), train_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    test_dataset = DownstreamLoader(os.path.join(root, "test"), test_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    val_dataset = DownstreamLoader(os.path.join(root, "val"), val_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_EEGMAT_dataset(root, name, contain_EEG=True, contain_EOG=False, contain_ECG=True, contain_EMG=False):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = DownstreamLoader(os.path.join(root, "train"), train_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    test_dataset = DownstreamLoader(os.path.join(root, "test"), test_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    val_dataset = DownstreamLoader(os.path.join(root, "val"), val_files, name, contain_EEG, contain_EOG, contain_ECG, contain_EMG)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def performance_metrics(Y, pre_Y, method):
    assert Y.shape == pre_Y.shape

    if method == 'r2':
        score = list(r2_score(Y, pre_Y, multioutput='raw_values'))
    elif method == 'rmse':
        score = list(root_mean_squared_error(Y, pre_Y, multioutput='raw_values'))
    elif method == 'pearsonr':
        score = [pearsonr(Y[:, idx],pre_Y[:,idx])[0] for idx in range(Y.shape[1])]
    else:
        print('Error! {} is undefined.'.format(method))
    score = sum(score) / len(score)
    return score


def get_metrics(output, target, metrics, is_binary):
    if 'r2' in metrics:
        results = {}
        for metric in metrics:
            results[metric] = performance_metrics(target, output, metric)
        return results
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results
