# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import inspect

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, num_hid),
            nn.Tanh()
        ) 
        self.layer2 = nn.Linear(num_hid,1)

    def forward(self, input):
        # convert to polar co-ordinates
        x = input[:,0]
        y = input[:,1]
        r = torch.sqrt((x*x) + (y*y))
        a = torch.atan2(y,x)
        output = torch.stack([r,a], dim=1)

        output = self.main(output)
        output = torch.sigmoid(self.layer2(output))
        return output
#'''
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, num_hid),
            nn.Tanh(),
            nn.Linear(num_hid,num_hid),
            nn.Tanh(),
            #nn.Linear(num_hid,num_hid),
            #nn.Tanh()
        )
   
        self.layer2 = nn.Linear(num_hid,1)

    def forward(self, input):
        output = self.main(input)
        output = torch.sigmoid(self.layer2(output))
        return output
#'''
'''
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # Changed to allow printing of first two layers
        self.main = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid,num_hid)
        self.layer3 = nn.Linear(num_hid,1)

    def forward(self, input):
        output = torch.tanh(self.main(input))
        output = torch.tanh(self.layer2(output))
        output = torch.sigmoid(self.layer3(output))
        return output
'''

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout

        # PolarNet
        netType = isinstance(net, PolarNet)
        if (netType):
            pgrid = []
            for [x,y] in pgrid:
                pgrid.append([math.sqrt(x*x + y*y), math.atan2(y,x)])
            pgrid = torch.tensor(grid)
            output = net.main(pgrid)

        # RawNet
        else:
            if layer >= 1:
                output = net.main(grid)
                output = torch.tanh(output)
            if layer >= 2:
                output = net.layer2(output)
                output = torch.tanh(output)

        net.train() # toggle batch norm, dropout back again
        pred = (output[:, node] >= 0).float()

        # plot function computed by model
        plt.clf()
        activation = pred.cpu().view(yrange.size()[0], xrange.size()[0])
        plt.pcolormesh(xrange,yrange,activation, cmap='Wistia')

