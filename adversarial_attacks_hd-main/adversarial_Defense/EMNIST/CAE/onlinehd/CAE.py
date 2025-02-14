import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Args:
    batch_size=128
    epochs=12
    no_cuda='store_true'
    seed=1234
    log_interval=10

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.X)

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.fc1 = nn.Linear(784, 10000, bias = False) # Encoder
        self.fc2 = nn.Linear(10000, 784, bias = False) # Decoder
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 784)))
        return h1

    def decoder(self,z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
            h1 = self.encoder(x)
            h2 = self.decoder(h1)
            return h1, h2
  
    def save_data(self, x, epoch, idx, recon): # save all preprocessed data
        _, samples = self.forward(x)
        id = idx*128
        
        samples = samples.data.cpu()
        for i, sample in enumerate(samples):
            recon[id+i]=samples[i]
        return recon