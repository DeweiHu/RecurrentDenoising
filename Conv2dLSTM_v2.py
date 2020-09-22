#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:40:12 2020

@author: hud4
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


#%% define the single 2d convolutional LSTM cell
'''
LSTM cell parameters
    input_dim: (int) Number of channels of input
    hidden_dim: (int) Number of channels of hidden state
    kernel_size: (tuple) 2d kernel
    bias: (bool) add bias or not
'''
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias):
        super(ConvLSTM, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias

        self.padding = int((kernel_size[0] - 1) / 2)
        
        # layer 1
        self.Wxi_1 = nn.Conv2d(self.input_channels, self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whi_1 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxf_1 = nn.Conv2d(self.input_channels, self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whf_1 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxc_1 = nn.Conv2d(self.input_channels, self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whc_1 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxo_1 = nn.Conv2d(self.input_channels, self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Who_1 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[0], self.kernel_size, 1, self.padding, bias=self.bias)
        
         # layer 2
        self.Wxi_2 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whi_2 = nn.Conv2d(self.hidden_channels[1], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxf_2 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whf_2 = nn.Conv2d(self.hidden_channels[1], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxc_2 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whc_2 = nn.Conv2d(self.hidden_channels[1], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxo_2 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)
        self.Who_2 = nn.Conv2d(self.hidden_channels[1], self.hidden_channels[1], self.kernel_size, 1, self.padding, bias=self.bias)

    def Cell_1(self, x, h, c): 
        ci = torch.sigmoid(self.Wxi_1(x) + self.Whi_1(h))
        cf = torch.sigmoid(self.Wxf_1(x) + self.Whf_1(h))
        cc = cf * c + ci * torch.tanh(self.Wxc_1(x) + self.Whc_1(h))
        co = torch.sigmoid(self.Wxo_1(x) + self.Who_1(h))
        ch = co * torch.tanh(cc)
        return ch, cc
    
    def Cell_2(self, x, h, c): 
        ci = torch.sigmoid(self.Wxi_2(x) + self.Whi_2(h))
        cf = torch.sigmoid(self.Wxf_2(x) + self.Whf_2(h))
        cc = cf * c + ci * torch.tanh(self.Wxc_2(x) + self.Whc_2(h))
        co = torch.sigmoid(self.Wxo_2(x) + self.Who_2(h))
        ch = co * torch.tanh(cc)
        return ch, cc

    def forward(self, input_tensor, hidden_channels):
        b, n_seq, n_ch, H, W = input_tensor.size()
        # initialize and overwrite
        h1 = Variable(torch.zeros(b, self.hidden_channels[0], H, W)).cuda() 
        c1 = Variable(torch.zeros(b, self.hidden_channels[0], H, W)).cuda() 
        h2 = Variable(torch.zeros(b, self.hidden_channels[1], H, W)).cuda() 
        c2 = Variable(torch.zeros(b, self.hidden_channels[1], H, W)).cuda()
        
        for i in range(n_seq):
            h1, c1 = self.Cell_1(input_tensor[:,i,:,:,:],h1,c1)
            h2, c2 = self.Cell_2(h1,h2,c2)
        
        return h2