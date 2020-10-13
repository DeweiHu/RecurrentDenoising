# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:09:57 2020

@author: hudew
"""

import torch
import torch.nn as nn

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
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
input_nch: [int] input tensor channel number
enc_nch: [tuple] output channel number of each resolution of the encoder 
dec_nch: [tuple] output channel number of each resolution of the decoder
device: cpu/gpu
'''

class lstm_UNet(nn.Module):
    def __init__(self, input_nch, enc_nch, device):
        super(lstm_UNet, self).__init__()
        
        self.input_nch = input_nch
        self.enc_nch = enc_nch
        self.device = device
        # set the kernel_size in LSTMcell to be fixed
        self.kernel_size = 3
        
        # define the single-layer ConvLSTMcells for image of each resolution
        cell_list = []
        for i in range(len(self.enc_nch)):
            cell_list.append(ConvLSTMcell(self.enc_nch[i], self.enc_nch[i],
                                          self.kernel_size, True))
        # add bottom layer
        cell_list.append(ConvLSTMcell(self.enc_nch[-1], self.enc_nch[-1],
                                      self.kernel_size, True))
        self.cell_list = nn.ModuleList(cell_list)
        
        # concatenate a list of channels for encoder & decoder
        self.nch_list = (self.input_nch,)+self.enc_nch
        
        # define the Res_branch and transition in encoder & decoder
        # for encoder/decoder[0] --> dual path
        #     encoder/decoder[1] --> single path
        #     transition[0] --> transition down
        #     transition[1] --> transition up
        encoder = [[]]*2
        decoder = [[]]*2
        transition = [[]]*2
        
        for i in range(len(self.enc_nch)):
            # encoder
            encoder[0].append(self.dual_branch(self.nch_list[i],self.nch_list[i+1]))
            encoder[1].append(self.single_branch(self.nch_list[i],self.nch_list[i+1]))
            transition[0].append(self.Transdown(self.enc_nch[i],self.enc_nch[i]))
            # decoder
            transition[1].append(self.Transup(self.enc_nch[-1-i],self.enc_nch[-1-i]))
            decoder[0].append(self.dual_branch(self.nch_list[-1-i]*2,self.nch_list[-2-i]))
            decoder[1].append(self.single_branch(self.nch_list[-1-i]*2,self.nch_list[-2-i]))
            
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.transition = nn.ModuleList(transition) 
    
    def forward(self, x, h_, c_):
        # encoder
        for i in range(len(self.enc_nch)):
            layer_opt = torch.add(self.encoder[0][i](x),self.encoder[1][i](x))
            x = self.transition[0][i](layer_opt)
            h_[i], c_[i] = self.cell_list[i](layer_opt,h_[i],c_[i])
        h_[-1], c_[-1] = self.cell_list[-1](x,h_[-1],c_[-1])
        
        # decoder
        layer_opt = h_[-1]
        for i in range(len(self.enc_nch)):
            x = self.transition[1][i](layer_opt)
            x = torch.cat([x, h_[-2-i]],dim=1)
            layer_opt = torch.add(self.decoder[0][i](x),self.decoder[1][i](x))
        
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
                nn.BatchNorm2d(num_features=in_ch),
                nn.ELU()
                )
