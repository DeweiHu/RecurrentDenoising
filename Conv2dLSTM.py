# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:46:51 2020

@author: hudew
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
"""
idea from
FULLY CONVOLUTIONAL STRUCTURED LSTM NETWORKS FOR JOINT 4D MEDICAL IMAGE SEGMENTATION
"""

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        #assert hidden_channels % 2 == 0
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))

        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        
        # front append the input-channel to list of hidden-channel
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # number of convLSTM cells stacked
        self.seq_dim = len(hidden_channels)
        
        # when step == 0, do initialization, step == 1, forward propagation
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        
        # define LSTM cells
        for i in range(self.seq_dim):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            
            # define attributes self.cell{} = cell
            setattr(self, name, cell)
            self._all_layers.append(cell)
            
    def forward(self, x):
        internal_state = []
        outputs = []
        for step in range(self.step+1):
            for i in range(self.seq_dim):
                # Initialize cells at the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    B, C, H, W = x.size()  # 2d input (B,C,H,W)
                    (h0, c0) = getattr(self, name).init_hidden(batch_size=B, hidden=self.hidden_channels[i],
                                                             shape=(H, W))
                    internal_state.append((h0, c0))
                    
                # Foward propagation
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)
        # output a tuple ()
        return outputs, (x, new_c)

#%% Model Check  
if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=1,hidden_channels=[32,64,64,32,1],kernel_size=3,step=1,
                        effective_step=[1]).cuda()
    criterion = torch.nn.MSELoss()

    x = Variable(torch.randn(1,1,64,64)).cuda()
    y = Variable(torch.randn(1,1,64,64)).double().cuda()

    y_hat = convlstm(x)
    y_hat = y_hat[0][0].double()
    
    res = torch.autograd.gradcheck(criterion,(y_hat,y),eps=1e-6,raise_exception=True)
