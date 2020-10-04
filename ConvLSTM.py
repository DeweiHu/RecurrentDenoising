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
    def __init__(self, nch_x, nch_h, kernel_size, bias, device):
        super(ConvLSTMcell, self).__init__()
        
        self.nch_x = nch_x
        self.nch_h = nch_h
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = int((self.kernel_size-1)/2)
        
        self.Wxi = nn.Conv2d(self.nch_x, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whi = nn.Conv2d(self.nch_h, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxf = nn.Conv2d(self.nch_x, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whf = nn.Conv2d(self.nch_h, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxc = nn.Conv2d(self.nch_x, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Whc = nn.Conv2d(self.nch_h, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)

        self.Wxo = nn.Conv2d(self.nch_x, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)
        self.Who = nn.Conv2d(self.nch_h, self.nch_h, self.kernel_size, 1, self.padding, bias=self.bias)
        
    def forward(self, x, h, c): 
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        return ch, cc

#%% Residual LSTM
class ResLSTM(nn.Module):
    def __init__(self, nlayer, nch_x, nch_h, kernel_size, bias, device):
        super(ResLSTM, self).__init__()
        # check layer number 
        if not len(kernel_size) == len(nch_h) == nlayer:
            raise ValueError('Inconsistent list length.')
        
        self.nlayer = nlayer
        self.nch_x = nch_x
        self.nch_h = nch_h
        self.kernel_size = kernel_size
        self.bias = bias
        self.device = device
        
        # len(nch_x, nch_h10, nch_h20, ..., nch_hn0)= n+1 
        self.dim_list = (self.nch_x,) + self.nch_h
        cell_list = []
        res_list = []
        for i in range(self.nlayer):
            cell_list.append(ConvLSTMcell(self.dim_list[i], self.dim_list[i+1],
                                          self.kernel_size[i], self.bias, device))
            res_list.append(nn.Conv2d(in_channels = self.dim_list[i],
                                      out_channels = self.dim_list[i+1],
                                      kernel_size = self.kernel_size[i],
                                      stride = 1,
                                      padding = int((self.kernel_size[i]-1)/2),
                                      bias = True)
                            )   
        self.cell_list = nn.ModuleList(cell_list)
        self.res_list = nn.ModuleList(res_list)
        
    # h_, c_ are tuples h_ = (h10,h20,h30,...,hn0)
    def forward(self, input_tensor):
        n_batch, n_seq, n_ch, H, W = input_tensor.shape
        
        # initialize states
        h_, c_ = self.init_state(input_tensor, self.nch_h, self.device)
        
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
                x = h + self.res_list[j](x)
            # update the state of all layers
            h_ = h_buff
            c_ = c_buff
        
        # the final output
        return x
        
    def init_state(self, input_tensor, nch_h, device):
        h_ = ()
        c_ = ()
        n_batch, n_seq, n_ch, H, W = input_tensor.shape
        
        # initialize with input 
        for i in range(len(nch_h)):
            h0 = torch.empty(n_batch,nch_h[i], H, W).to(device)
            c0 = torch.empty(n_batch,nch_h[i], H, W).to(device)
            for j in range(nch_h[i]):
                h0[:,j,:,:] = input_tensor[:,0,:,:,:]
                c0[:,j,:,:] = torch.mean(input_tensor,dim=1,keepdim=True)
            h_ = h_ + (h0,)
            c_ = c_ + (c0,)
        return h_, c_
        
        