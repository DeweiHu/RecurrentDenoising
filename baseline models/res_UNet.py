# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 04:51:00 2020

@author: hudew
"""

import torch 
import torch.nn as nn

class res_UNet(nn.Module):
    def __init__(self, input_nch, enc_nch):
        super(res_UNet, self).__init__()
        
        self.enc_nch = enc_nch
        self.input_nch = input_nch
        self.kernel_size = 3
        self.res_nch = (self.input_nch,)+self.enc_nch
        
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
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.enc_nch)):
            layer_opt = self.relu(torch.add(self.encoder_dual[i](x),self.encoder_single[i](x)))
            x = self.transition_down[i](layer_opt)
            cats.append(layer_opt)
        
        layer_opt = x
        for i in range(len(self.enc_nch)):
            x = self.transition_up[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.relu(torch.add(self.decoder_dual[i](x),self.decoder_single[i](x)))
        
        y_pred = layer_opt
        return y_pred
        
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
