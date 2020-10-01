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
nlayer: [int] cell layers
nch_x: [int] number of channels of input image
nch_h: [tuple] a tuple that contains number of channels of every layer
kernal_size: [tuple] a tuple that contains kernel size of each layer
bias: [boolean]  
'''
class LSTMModel(nn.Module):
    def __init__(self, nlayer, nch_x, nch_h, kernel_size, bias):
        super(LSTMModel, self).__init__()
        # check layer number 
        if not len(kernel_size) == len(nch_h) == nlayer:
            raise ValueError('Inconsistent list length.')
        
        self.nlayer = nlayer
        self.nch_x = nch_x
        self.nch_h = nch_h
        self.kernel_size = kernel_size
        self.bias = bias
        
        # len(nch_x, nch_h10, nch_h20, ..., nch_hn0)= n+1 
        self.dim_list = (self.nch_x,) + self.nch_h
        cell_list = []
        for i in range(self.nlayer):
            self.cell_list.append(ConvLSTMcell(self.dim_list[i], self.dim_list[i+1],
                                               self.kernel_size[i], self.bias))    
        self.cell_list = nn.ModuleList(cell_list)
    
    # h_, c_ are tuples h_ = (h10,h20,h30,...,hn0)
    def forward(self, input_tensor, h_, c_):
        n_batch, n_seq, n_ch, H, W = input_tensor.shape
        # state buffer
        h_buff = ()
        c_buff = ()
        
        # iter over sequence
        for i in range(n_seq):
            x = input_tensor[:,i,:,:,:]
            # iter over layers
            for j in range(len(self.cell_list)):
                h, c = self.cell_list[j](x, h_[j], c_[j])
                # save the hidden and cell state in a buffer
                h_buff = h_buff + (h,)
                c_buff = c_buff + (c,)
                # update the input
                x = h
            # update the state of all layers
            h_ = h_buff
            c_ = c_buff
        
        # the final output
        return h_[-1]
    
    def init_state(self, nch_h):
        h_ = ()
        c_ = ()
        for i in range(len(nch_h)):
            
        
        
        
        