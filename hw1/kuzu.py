# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.lin = nn.Linear(784,10)

    def forward(self, x):
        x = x.view(-1,784) # reshape to have 784 columns  
        x = F.log_softmax(self.lin(x), dim=1)
        return x 

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.hiddennodes = 100
        self.main = nn.Sequential(
            nn.Linear(784, self.hiddennodes),
            nn.Tanh(),
            nn.Linear(self.hiddennodes, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1,784)
        x = self.main(x)
        return x
'''
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer1 = nn.Linear(2304,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0),-1)
        x = self.layer1(x)
        x = F.log_softmax(x, dim=1)
        return x
'''
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2)
        )
        self.layer1 = nn.Linear(2304,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0),-1)
        x = self.layer1(x)
        x = F.log_softmax(x, dim=1)
        return x
