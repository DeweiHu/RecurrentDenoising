# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:19:03 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\RNN\\')
import util
import UNet_LSTM
import res_UNet

import pickle,time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

dataroot = 'E:\\HumanData\\UNet+LSTM_train.pickle'
modelroot = 'E:\\Model\\'

enc_nch_1 = (1,16,32,64)
enc_nch_2 = (4,16,32)
n_seq = 5
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = UNet_LSTM.lstm_UNet(enc_nch_1).to(device)
model1.load_state_dict(torch.load(modelroot+'UNet_LSTM_L5_seq=5.pt'))

model2 = res_UNet.res_UNet(n_seq,enc_nch_2).to(device)
model2.load_state_dict(torch.load(modelroot+'res_UNet_L4.pt'))

# x:[batch,n_seq,n_channel,H,W], [0,255]
# y:[batch,n_seq, n_channel,H,W], [0,1]
class MyDataset(Data.Dataset):

    def ToTensor(self, x, y):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=1)  # add channel dimension
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        y_tensor = torch.unsqueeze(y_tensor,dim=1)
        return x_tensor, y_tensor    
        
    def __init__(self, dir):
       with open(dir,'rb') as f:
           self.pair = pickle.load(f)
        
    def __len__(self):
        return len(self.pair)

    def __getitem__(self,idx):
        x, y = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(x,util.ImageRescale(y,[0,1]))
        return x_tensor, y_tensor

test_loader = Data.DataLoader(dataset=MyDataset(dataroot),batch_size=batch_size, shuffle=False)

#%% Test
for step,(x_seq,y_seq) in enumerate(test_loader):
    with torch.no_grad():
        h_ = []
        c_ = []
        B, n_seq, _, H, W = x_seq.size()
        h_, c_ = UNet_LSTM.state_init(enc_nch_1,B,H,W,device)
        
        for i in range(n_seq):
            x1 = Variable(x_seq[:,i,:,:,:]).to(device)    
            y_pred_1, h_, c_ = model1(x1, h_, c_)
        
        x2 = Variable(torch.squeeze(x_seq,dim=1)).to(device)
        y_pred_2 = model2(x2)
        
        im_x = util.ImageRescale(x2[0,-1,:,:500].detach().cpu().numpy(),[0,255])
        im_y = util.ImageRescale(y_seq[0,0,:,:500].numpy(),[0,255])
        im_1 = util.ImageRescale(y_pred_1[0,0,:,:500].detach().cpu().numpy(),[0,255])
        im_2 = util.ImageRescale(y_pred_2[0,0,:,:500].detach().cpu().numpy(),[0,255])
        
        bot = np.concatenate((im_x,im_y),axis=1)
        top = np.concatenate((im_1,im_2),axis=1)

        if step % 100 == 0:
            plt.figure(figsize=(16,16))
            plt.axis('off')
            plt.title('Slice: {}'.format(step),fontsize=20)
            plt.imshow(np.concatenate((top,bot),axis=0),cmap='gray')
            plt.show()
