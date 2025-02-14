import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import tensorflow as tf
import os
import pickle
import copy
from loader import get_dataset


x, y = get_dataset()  

net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()
torch.manual_seed(1234)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def recover_image(x):
    img = (x + 0.5)*255
    img = Image.fromarray(img).convert('RGB')
    return img

im_orig = recover_image(x.numpy()[0][0])

im = transforms.Compose([
    transforms.Scale(256),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                            std = std)])(im_orig)

pert_image = deepfool(im, net)

clip = lambda x: clip_tensor(x, 0, 1)
tf = transforms.Compose([transforms.Normalize(mean=[0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1]),
                        transforms.Lambda(clip),
                        transforms.Grayscale(1)])

attack_image = tf(pert_image.cpu()[0])

transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                            std = std)])

for i in range(1, len(x)):
    im_orig = recover_image(x.numpy()[i][0])
    im = transforms(im_orig)
    pert_image = deepfool(im, net)
    pert_image = tf(pert_image.cpu()[0])

    attack_image = torch.cat([attack_image, pert_image], dim=0)

# save
with open('MNIST_DeepFool.pickle', 'wb') as f:
    pickle.dump(attack_image, f, pickle.HIGHEST_PROTOCOL)
