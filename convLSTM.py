# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:44:18 2020

@author: hudew
"""

import torch
import torch.nn as nn

'''
LSTM cell parameters
    input_dim: int. Number of channels of input
    hidden_dim: int. Number of channels of hidden state
    bias: bool. add bias or not
'''

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = 1
        self.bias = bias
        
        # in_channels: concatenate of h_{t-1} and x_{t}
        # out_channels: f,i,g,o (4 gates)
        self.conv = nn.Conv2d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = 4*self.hidden_dim,
                              kernel_size = self.kernel_size,
                              padding = self.padding,
                              bias = self.bias)
        
    # x: [batch,channel,n_seq,H,W]
    def foward(self, x, ipt_state):
        ipt_h, ipt_c = ipt_state
        combined = torch.cat([x,ipt_h],dim=1) #concatenate along channel
        
        # get concatenated 4 gates
        combined_conv = self.conv(combined)
        # split 4 gates in channel and change domain
        gate_i,gate_f,gate_o,gate_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_o = torch.sigmoid(gate_o)
        gate_g = torch.tanh(gate_g)
        
        opt_c = gate_f*ipt_c + gate_i*gate_g
        opt_h = gate_o*torch.tanh(opt_c)
        
        return opt_h, opt_c
        
    def init_hidden(self, batch_size, image_size):
        H,W = image_size
        return(torch.zeros(batch_size, self.hidden_dim, H, W).cuda(),
               torch.zeros(batch_size, self.hidden_dim, H, W).cuda())

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
        
        # check the form of kernel_size 
        # condition 1: kernel_size is a tuple, (3,3)
        # condition 2: kernel_size is a list of tuple
        @staticmethod
        def _check_kernel_size_consistency(kernel_size):
            if not (isinstance(kernel_size, tuple) or \
                (isinstance(kernel_size, list) \
                and all([isinstance(elem, tuple) for elem in kernel_size]))):
                raise ValueError('"kernel_size" must be tuple or list of tuples')
        
        # if variable is not yet a list, then make it a list of tuple
        @staticmethod
        def _extand_for_multilayer(param, num_layers):
            if not isinstance(param,list):
                param = [param]*num_layers
            return param
                
        self._check_kernel_size_consistency(kernel_size)
        
        kernel_size = self._extend_for_multilayer(kernel_size,num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim,num_layers)
    
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        
        # vertical (if num_layers=1, only 1 type of cell is defined)
        for i in range(self.num_layers):
            ipt_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            
            cell_list.append(ConvLSTMCell(input_dim = ipt_dim,
                                          hidden_dim = self.hidden_dim[i],
                                          kernel_size = self.kernel_size[i],
                                          bias = self.bias))
            self.cell_list = nn.ModuleList(cell_list)
       
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size,image_size))
        return init_states
    
    def forward(self, x, hidden_state=None):
        # if input is (t,b,c,h,w), make it (b,t,c,h,w)
        if not self.batch_first:
            x = x.permute(1,0,2,3,4)
        
        batchsize,seq_len,n_ch,H,W = x.size()
        
        # Initialize with h0,c0: [batch_size,hidden_dim,H,W]
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=batchsize,image_size=(H,W))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = x
        
        # propagate through LSTM layers
        for layer_idx in range(self.num_layers):
            h,c = hidden_state[layer_idx]
            inter_h = []
            
            # propagate through sequence
            for t in range(seq_len):
                h,c = self.cell_list[layer_idx](x = cur_layer_input[:,t,:,:,:],
                                                itp_state = [h,c])
                inter_h.append(h)
            
            layer_output = torch.stack(inter_h,dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h,c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
            
                
                
                
        
                
        


