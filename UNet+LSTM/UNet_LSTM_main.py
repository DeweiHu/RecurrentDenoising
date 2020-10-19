#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 05:02:38 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/2020/RNN/')
import util
import UNet_LSTM as arch

import pickle,time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

dataroot = '/home/hud4/Desktop/2020/Data/UNet+LSTM_ONH.pickle'
modelroot = '/home/hud4/Desktop/2020/Model/'

batch_size = 1
enc_nch = (1,16,32,64)
n_seq = 5
n_epoch = 100
criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()
alpha = 5
beta = 2
learning_rate = 0.0001
epoch_loss = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = arch.lstm_UNet(enc_nch).to(device)
model.load_state_dict(torch.load(modelroot+'UNet_LSTM_L5_seq=5.pt'))

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

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

train_loader = Data.DataLoader(dataset=MyDataset(dataroot),batch_size=batch_size, shuffle=True)

#%% dimension check
#for step,(x,y) in enumerate(train_loader):
#    pass
#print(step)
#print(x.size())
#print(y.size())

#%% train
t1 = time.time()

for epoch in range(n_epoch):
    sum_loss = 0
    for step,(x_seq,y_seq) in enumerate(train_loader):
        model.train()
        h_ = []
        c_ = []
        # initialize the state
        B, n_seq, _, H, W = x_seq.size()
        h_, c_ = arch.state_init(enc_nch,B,H,W,device)
        
        # iterate over sequence, every tp supervised
        for i in range(n_seq):
            x = Variable(x_seq[:,i,:,:,:]).to(device)
            y = Variable(y_seq[:,i,:,:,:]).to(device)    
            y_pred, h_, c_ = model(x, h_, c_)
            
            loss = alpha*criterion1(y_pred, y)+beta*criterion2(y_pred, y)
            sum_loss += loss
            epoch_loss.append(sum_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if step % 200 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'%(epoch,n_epoch,step,len(train_loader),loss.item()))
            
        if step == len(train_loader)-1:
            pred = util.ImageRescale(y_pred[0,0,:,:500].detach().cpu().numpy(),[0,255])
            im_y = util.ImageRescale(y[0,0,:,:500].detach().cpu().numpy(),[0,255])
            im_x = util.ImageRescale(x[0,0,:,:500].detach().cpu().numpy(),[0,255])
            im_h = util.ImageRescale(h_[0][0,0,:,:500].detach().cpu().numpy(),[0,255])
            
            top = np.concatenate((im_x,im_h),axis=1)
            bot = np.concatenate((pred,im_y),axis=1)
            
            plt.figure(figsize=(16,16))
            plt.axis('off')
            plt.title('Epoch: {}'.format(epoch),fontsize=20)
            plt.imshow(np.concatenate((top,bot),axis=0),cmap='gray')
            plt.show()

t2 = time.time()
print('Training finished. Time: {} min'.format((t2-t1)/60))

#%%
name = 'UNet_LSTM_L5_seq=5.pt'
torch.save(model.state_dict(),modelroot+name)

#%%
plt.figure()
plt.title('Loss vs. Epoch',fontsize=20)
plt.plot(epoch_loss)
plt.show()