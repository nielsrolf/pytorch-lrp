import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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

        return {self.classes[i]: class_correct[i]/class_total[i] for i in range(10)}

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir)

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir))
        

def one_hot(labels):
    return torch.from_numpy(np.eye(10)[labels]).float()


def input_times_gradient(net, images, target_pattern):
    # pattern: eg one_hot(labels)
    img = torch.tensor(images, requires_grad=True)
    v = (net.forward(img)*target_pattern).sum()
    v.backward()
    h = img.data*img.grad
    return h


def to_numpy(x):
    if isinstance(x, np.ndarray): return x
    return x.detach().numpy()