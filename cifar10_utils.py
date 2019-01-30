#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class TrainableNet(nn.Module):
    def train(self, epochs, trainloader):
        net = self
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    
    def accuracy(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct/total
    
    def class_accuracy(self, testloader):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        return {classes[i]: class_correct[i]/class_total[i] for i in range(10)}

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir)

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir))


class Net(TrainableNet):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    


def one_hot(labels):
    return torch.from_numpy(np.eye(10)[labels]).float()


def input_times_gradient(net, images, target_pattern):
    # pattern: eg one_hot(labels)
    img = torch.tensor(images, requires_grad=True)
    v = (net.forward(img)*target_pattern).sum()
    v.backward()
    h = (img.data*img.grad).sum(1)
    return h


def to_numpy(x):
    if isinstance(x, np.ndarray): return x
    return x.detach().numpy()


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



