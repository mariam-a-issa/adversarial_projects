import os
import pickle

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import torch
from tensorflow.keras.datasets import fashion_mnist
from torchvision.datasets import EMNIST


def normalize(x_train, x_test, type='sklearn'):
    if type is None or type.lower() == 'none':
        return x_train, x_test
    elif type.lower() == 'sklearn':
        scaler = sklearn.preprocessing.Normalizer().fit(x_train)
        normalized_x_train = scaler.transform(x_train)
        normalized_x_test = scaler.transform(x_test)
    elif type.lower() == 'minmax':
        min_ = min(np.min(x_train), np.min(x_test))
        max_ = max(np.max(x_train), np.max(x_test))
        normalized_x_train = (x_train - min_) / (max_ - min_)
        normalized_x_test = (x_test - min_) / (max_ - min_)
    elif type.lower() == '255':
        normalized_x_train = x_train / 255.
        normalized_x_test = x_test / 255.
    else:
        print('Unknown normalize type: {}'.format(type))
        raise AssertionError

    return normalized_x_train, normalized_x_test


def load_mnist():
    x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    x = np.array(x).astype(np.float64)
    y = np.array(y).astype(np.int64)

    # Split
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)

    x_train = x_train.reshape(-1, 28, 28)
    y_train = y_train.flatten()
    x_test = x_test.reshape(-1, 28, 28)
    y_test = y_test.flatten()

    return x_train, y_train, x_test, y_test, scaler


def load_emnist():
    temp = EMNIST('./data/EMNIST', split='letters', train=True, download=True)
    x_train = temp.data.numpy().transpose((0, 2, 1))
    y_train = temp.targets.numpy() - 1

    temp = EMNIST('./data/EMNIST', split='letters', train=False, download=True)
    x_test = temp.data.numpy().transpose((0, 2, 1))
    y_test = temp.targets.numpy() - 1

    return x_train, y_train, x_test, y_test


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    return x_train, y_train, x_test, y_test


def load_pickle_data(load_path):
    with open(load_path, 'rb') as handle:
        data = pickle.load(handle)
        x_train = data['train_data'].numpy()
        y_train = data['train_label'].numpy()
        x_test = data['test_data'].numpy()
        y_test = data['test_label'].numpy()

    return x_train, y_train, x_test, y_test


def save_pickle_data(x_train, y_train, x_test, y_test, save_path):
    data = {
        'train_data': torch.from_numpy(x_train),
        'train_label': torch.from_numpy(y_train),
        'test_data': torch.from_numpy(x_test),
        'test_label': torch.from_numpy(y_test),
    }

    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(dataset, data_dir, print_info=True, save_data=True, use_new=False):
    print('Loading {} Data'.format(dataset))

    dataset_dir = os.path.join(data_dir, dataset)
    pickle_path = os.path.join(dataset_dir, '{}.pickle'.format(dataset))

    if os.path.isfile(pickle_path) and not use_new:
        x_train, y_train, x_test, y_test = load_pickle_data(pickle_path)
    else:
        if dataset.lower() == 'fmnist':
            x_train, y_train, x_test, y_test = load_fashion_mnist()
        elif dataset.lower() == 'emnist':
            x_train, y_train, x_test, y_test = load_emnist()
        elif dataset.lower() == 'mnist':
            x_train, y_train, x_test, y_test = load_mnist()
        else:
            print('Unknown dataset: {}'.format(dataset))
            raise AssertionError

        # Save data
        if save_data:
            os.makedirs(dataset_dir, exist_ok=True)
            save_pickle_data(x_train, y_train, x_test, y_test, pickle_path)

    x_train, x_test = normalize(x_train, x_test, type='minmax')
    min_ = min(np.min(x_train), np.min(x_test))
    max_ = max(np.max(x_train), np.max(x_test))
    classes = np.unique([np.unique(y_train.flatten()), np.unique(y_test.flatten())])
    num_classes = len(classes)

    if print_info:
        print('Train dataset Size:', list(x_train.shape), list(y_train.shape))
        print('Train dataset Labels:', np.unique(y_train))
        print('Test dataset Size:', list(x_test.shape), list(y_test.shape))
        print('Test dataset Labels:', np.unique(y_test))
        print('Data value: {} ~ {}'.format(min_, max_))
        print('Num classes: {}'.format(num_classes))

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    return (x_train, y_train), (x_test, y_test), min_, max_, num_classes
