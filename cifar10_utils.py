#!/usr/bin/env python
# coding: utf-8

"""
The code is mostly a densed version of the pytorch cifar 10 tutorial here https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-download-beginner-blitz-cifar10-tutorial-py
"""

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trainable_net import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class Net(TrainableNet):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.classes = classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


def plot_heatmaps(images, heatmaps):
    images, heatmaps = to_numpy(images), to_numpy(heatmaps)
    print(images.shape)
    images = images.transpose([0,2,3,1])
    for img, h in zip(images, heatmaps):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
                
        ax1.imshow((img+1)/2)
        ax1.axis('off')
        
        H = 0*img + 1
        r = np.maximum(h, 0)*3
        b = np.maximum(-h, 0)*3
        H[:,:,[1,2]] -= r[...,None]
        H[:,:,[0,1]] -= b[...,None]
        ax2.imshow(H)
        ax2.axis('off')
        
        merge = img*0.25 + 0.5
        merge[:,:,0] += r
        merge[:,:,2] += b
        ax3.imshow(merge)
        ax3.axis('off')
        
        plt.show()



