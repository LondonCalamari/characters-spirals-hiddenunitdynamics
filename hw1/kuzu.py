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
        self.hiddennodes = 90
        self.h1 = nn.Linear(784, self.hiddennodes)
        self.h2 = nn.Linear(self.hiddennodes, 10)

    def forward(self, x):
        x = x.view(-1,784)
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        x = F.log_softmax(x, dim=1)
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        return 0 # CHANGE CODE HERE
