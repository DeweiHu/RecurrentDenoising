#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:15:01 2020

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
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # in_channels: concatenate of h_{t-1} and x_{t}
        # out_channels: f,i,g,o (4 gates)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
     # x: [batch,channel,n_seq,H,W]
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        # get concatenated 4 gates
        combined_conv = self.conv(combined)
        
        # split 4 gates in channel and change domain
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

#%%  Construct layer & sequence of LSTM  
'''
Model parameters:
    input_dim: (int) Number of channels in input
    hidden_dim: (int) Number of hidden channels
    kernel_size: (tuple/list of tuple) Size of kernel in convolutions
    num_layers: (int) Number of LSTM layers stacked 
    batch_first: (bool) input dimension 0 is batchsize or not
    bias: (bool) bias for convolution
    return_all_layers: (bool) Return the list of computations for all layers
Input: [batchsize,sequence_length,channel,H,W]/
       [sequence_length,batchsize,channel,H,W]
Output: A tuple of two lists with length num_layers/1
        0 - layer_output_list: [[h1,h2,...,hseq],[h1',h2',...,hseq'],...]
        1 - last_state_list: [(hseq,cseq),(hseq',cseq'),...]
'''
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
             # if input is (t,b,c,h,w), make it (b,t,c,h,w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Initialize with h0,c0: [batch_size,hidden_dim,H,W]
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        # propagate through LSTM layers
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # propagate through sequence
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    # check the form of kernel_size 
    # condition 1: kernel_size is a tuple, (3,3)
    # condition 2: kernel_size is a list of tuple
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
            
    # if variable is not yet a list, then make it a list of tuple
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
#%% overall model
class DN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(DN_LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        self.LSTM_pre = ConvLSTM(self.input_dim,self.hidden_dim,self.kernel_size,self.num_layers,
                                 self.num_layers,self.batch_first,self.bias,self.return_all_layers)
        
        self.LSTM_post = ConvLSTM(self.input_dim,self.hidden_dim,self.kernel_size,self.num_layers,
                                 self.num_layers,self.batch_first,self.bias,self.return_all_layers)
    
        self.fuse = nn.sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.ReLU()
                )
        
    def reverse_sequence(self, input_tensor):
        dim = list(input_tensor.size())
        
        if not len(dim) == 5:
            raise Exception('Input dimension should be 5, now it is {}'.format(len(5)))
        
        opt = torch.zeros(dim,dtype=torch.float32)
        for i in range(dim[1]):
            opt[:,i,:,:,:] = input_tensor[:,dim[1]-1-i,:,:,:]
        return opt       
    
    # X: [n_batch, n_seq, n_ch, H, W] 
    # hardcode: r=2
    def forward(self, X):
        x_pre = X[:,:3,:,:,:]
        x_post = self.reverse_sequence(X[:,2:,:,:,:])
        
        _, hc_pre = self.LSTM_pre(x_pre)
        _, hc_post = self.LSTM_post(x_post)
        h_pre = hc_pre[0][0]
        h_post = hc_post[0][0]
        
        y_hat = self.fuse(torch.cat((h_pre,h_post),dim=1))
        
        return h_pre,h_post,y_hat