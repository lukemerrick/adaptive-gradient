import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = [nn.Linear(layer_sizes[i], layer_sizes[i+1])\
                            for i in range(len(layer_sizes)-1)]
        # initialize weights
        for i, layer in enumerate(self.layers):
            layer.weight = nn.init.xavier_normal(layer.weight,
                                                gain=nn.init.calculate_gain('relu'))
            layer_name = 'output layer' if i == len(self.layers)-1\
                                else 'hidden layer {}'.format(i+1)
            self.add_module(layer_name, layer)

    def forward(self, x):
        # flatten 2d input
        x = x.view(-1, self.layer_sizes[0])
        # run through hidden layers
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        # run through final non-relu layer
        return F.softmax(self.layers[-1](x), dim=-1)
