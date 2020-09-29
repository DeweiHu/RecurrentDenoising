#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:13:47 2020

@author: hud4
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

#%% define a LSTM cell
'''
nch_x: [int] number of channels of input x
nch_h: [int] number of channels of hidden state h
kernel_size: [int] square kernel by default
bias: [boolean] add bias or not
'''
class ConvLSTMcell(nn.Module):
    def __init__(self, nch_x, nch_h, kernel_size, bias):
        super(ConvLSTMcell, self).__init__()
        
        self.nch_x = nch_x
        self.nch_h = nch_h
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = int((self.kernel_size-1)/2)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=self.bias)
        
    def forward(self, x, h, c): 
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        return ch, cc

'''

'''
class 