# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:09:57 2020

@author: hudew
"""

import torch
import torch.nn as nn

# h,c: [batch, n_channel, H, W]
def state_init(n_ch,batch_size,H,W,device):
    h_ = []
    c_ = []
    nch = n_ch + n_ch[-1:]
    for i in range(len(nch)):
        h_.append(torch.zeros(batch_size,nch[i],int(H/2**i),int(W/2**i)).to(device))
        c_.append(torch.zeros(batch_size,nch[i],int(H/2**i),int(W/2**i)).to(device))
    return h_, c_
        
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

'''
enc_nch: [tuple] output channel number of each resolution of the encoder 

1. In this architecture, there is no residual unit at the bottom layer, 
this can have some effect on how the network is defined.
2. The channel number needs to be changed in residual unit, so it is not 
strictly residual. In order to keep the channel number consistant, one path 
has two convolution block, another has one.
3. The downsample/upsample is realized by convolution (transition down/up)
4. The inisialization of cell state and hidden state need to be done outside
5. The input channel = output channel for ConvLSTMcell
'''

class lstm_UNet(nn.Module):
    def __init__(self, enc_nch):
        super(lstm_UNet, self).__init__()
        
        self.enc_nch = enc_nch
        self.input_nch = 1    # fixed input channel
        self.kernel_size = 3  # fixed kernel_size
        
        # channel list for residual unit (input,l1,l2,l3)
        self.res_nch = (self.input_nch,)+self.enc_nch
        # channel list for ConvLSTMcell (l1,l2,l3,l4)
        self.lstm_nch = self.enc_nch + self.enc_nch[-1:]
        
        # define the single-layer ConvLSTMcells for image of each resolution
        cell_list = []
        for i in range(len(self.lstm_nch)):
            cell_list.append(ConvLSTMcell(self.lstm_nch[i], self.lstm_nch[i],
                                          self.kernel_size, True))
        self.cell_list = nn.ModuleList(cell_list)

        # define the Res_branch and transition in encoder & decoder
        encoder_dual = []
        encoder_single = []
        decoder_dual = []
        decoder_single = []
        transition_up = []
        transition_down = []
        
        for i in range(len(self.enc_nch)):
            # encoder
            encoder_dual.append(self.dual_branch(self.res_nch[i],self.res_nch[i+1]))
            encoder_single.append(self.single_branch(self.res_nch[i],self.res_nch[i+1]))
            transition_down.append(self.Transdown(self.enc_nch[i],self.enc_nch[i]))
            # decoder
            transition_up.append(self.Transup(self.enc_nch[-1-i],self.enc_nch[-1-i]))
            decoder_dual.append(self.dual_branch(self.res_nch[-1-i]*2,self.res_nch[-2-i]))
            decoder_single.append(self.single_branch(self.res_nch[-1-i]*2,self.res_nch[-2-i]))
            
        self.encoder_dual = nn.ModuleList(encoder_dual)
        self.encoder_single = nn.ModuleList(encoder_single)
        self.decoder_dual = nn.ModuleList(decoder_dual)
        self.decoder_single = nn.ModuleList(decoder_single)
        self.transition_down = nn.ModuleList(transition_down) 
        self.transition_up = nn.ModuleList(transition_up)
        
        self.relu = nn.ReLU()
    
    # h_,c_ are list of tensors
    def forward(self, x, h_, c_):
        
        if not len(h_) == len(c_) ==len(self.lstm_nch):
            raise ValueError('missing state.')
        
        # encoder
        for i in range(len(self.enc_nch)):
            layer_opt = self.relu(torch.add(self.encoder_dual[i](x),self.encoder_single[i](x)))
            x = self.transition_down[i](layer_opt)
            h_[i], c_[i] = self.cell_list[i](layer_opt,h_[i],c_[i])
        
        # bottom layer
        h_[-1], c_[-1] = self.cell_list[-1](x,h_[-1],c_[-1])
        layer_opt = h_[-1]
        
        # decoder
        for i in range(len(self.enc_nch)):
            x = self.transition_up[i](layer_opt)
            x = torch.cat([x, h_[-2-i]],dim=1)
            layer_opt = torch.add(self.decoder_dual[i](x),self.decoder_single[i](x))
        
        y_pred = layer_opt
        
        return y_pred, h_, c_ 
        
    def Transdown(self,in_ch,out_ch):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, 
                          out_channels=out_ch, 
                          kernel_size=4,
                          stride=2, 
                          padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU()
                )
        
    def Transup(self,in_ch,out_ch):
        return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, 
                                   out_channels=out_ch,
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU()
                )
    
    def dual_branch(self,in_ch,out_ch):
        return nn.Sequential(
                nn.Conv2d(in_channels = in_ch,
                          out_channels = in_ch,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features=in_ch),
                nn.ELU(),
                nn.Conv2d(in_channels = in_ch,
                          out_channels = out_ch,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features=out_ch)
                )
    
    def single_branch(self, in_ch, out_ch):
        return nn.Sequential(
                nn.Conv2d(in_channels = in_ch,
                          out_channels = out_ch,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features=out_ch),
                nn.ELU()
                )
