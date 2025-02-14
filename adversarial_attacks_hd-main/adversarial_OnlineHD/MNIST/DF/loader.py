import torchvision
import numpy as np
import torch
import tensorflow as tf

def get_dataset():
    dataset = 'mnist' # dataset can be 'fashion_mnist', 'mnist', or 'emnist'
    mnist = tf.keras.datasets.mnist
    fashion_mnist = tf.keras.datasets.fashion_mnist
    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()
    else:
        temp = torchvision.datasets.EMNIST('./data/EMNIST', split = 'letters', train = True, download = True)
        x = temp.data.unsqueeze(3).numpy().transpose((0,2,1,3))
        y = temp.targets.numpy() - 1
        temp = torchvision.datasets.EMNIST('./data/EMNIST', split = 'letters', train = False, download = True)
        x_test = temp.data.unsqueeze(3).numpy().transpose((0,2,1,3))
        y_test = temp.targets.numpy() - 1
    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long().squeeze()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long().squeeze()

    x = torch.cat([x, x_test], dim=0) 
    y = torch.cat([y, y_test], dim=0) 
    if len(x.shape) == 3:
        x = x.unsqueeze(3)
        x_test = x_test.unsqueeze(3)
    return x, y  

x, y = get_dataset()

# save
with open('MNIST_DeepFool_label.pickle', 'wb') as f:
    pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)