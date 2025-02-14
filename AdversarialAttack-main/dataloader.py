import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.model_selection import train_test_split

import numpy as np
import struct

import matplotlib.pyplot as plt
import scipy.stats as st

import onlinehd
import spatial


def load_choirdat(dataset_path, train_data=None, train_label=None):
    with open(dataset_path, 'rb') as f:
        # reads meta information
        features = struct.unpack('i', f.read(4))[0]
        classes = struct.unpack('i', f.read(4))[0]
        
        # lists containing all samples and labels to be returned
        samples = list()
        labels = list()

        while True:
            # load a new sample
            sample = list()

            # load sample's features
            for i in range(features):
                val = f.read(4)
                if val is None or not len(val):

                    return (samples, labels), features, classes
                sample.append(struct.unpack('f', val)[0])

            # add the new sample and its label
            label = struct.unpack('i', f.read(4))[0]
            if train_data==None:
                samples.append(sample)
                labels.append(label)
            else:
                train_data.append(sample)
                train_label.append(label)
                
                
    
    return (samples, labels), features, classes

# loads simple mnist dataset
def load(dataset_name=None, sklearn_dataset_name='mnist_784'):
    # fetches data
    
    if dataset_name is None:
        x, y = sklearn.datasets.fetch_openml(sklearn_dataset_name, return_X_y=True)
        x = x.astype(np.float32)
        y = y.astype(np.int32)

        x=np.array(x)
        y=np.array(y)

        # split and normalize
        x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y)
        scaler = sklearn.preprocessing.Normalizer().fit(x)
        x = scaler.transform(x)
        x_test = scaler.transform(x_test)

        # changes data to pytorch's tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).long()
        return x, x_test, y, y_test
    
    
    train_path = 'dataset/' + dataset_name + '/' + dataset_name + '_train.choir_dat'
    test_path = 'dataset/' + dataset_name + '/' + dataset_name + '_test.choir_dat'

    (train_x, train_y), features, classes = load_choirdat(train_path)
    (test_x, test_y), features, classes = load_choirdat(test_path)
    
    scaler = sklearn.preprocessing.Normalizer().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
        
    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.LongTensor(test_y)
    
    return train_x, test_x, train_y, test_y
    
