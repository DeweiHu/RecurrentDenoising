#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:15:32 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/2020/RNN/')
import Conv2dLSTM
import util

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

dataroot = '/home/hud4/Desktop/2020/Data/'

# Model
input_ch = 1
hidden_ch = [4,8,16,32,1]
kernel_size = (3,3)
num_layers = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Conv2dLSTM.DN_LSTM(input_ch,hidden_ch,kernel_size,num_layers,True,True,False)
model = model.to(device)

#check parameter number
#for i in range(len(list(model.parameters()))):
#    print(list(model.parameters())[i].size())

#%% DataLoader
batch_size = 1

# x:[batch,seq_dim,channel,H,W]
# y:[batch,channel,H,W]
class MyDataset(Data.Dataset):

    def ToTensor(self, x, y):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=1)  # add channel dimension
        y_tensor = transforms.functional.to_tensor(y)
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
    
train_loader = Data.DataLoader(dataset=MyDataset(dataroot+'train.pickle'), 
                               batch_size=batch_size, shuffle=True)

# dimension check
#for step,(x,y) in enumerate(train_loader):
#    pass
#print(x.size())
#print(y.size())
    
#%% loss
num_epoch = 50

criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()

learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
r = 2

sum_loss = []

t1 = time.time()

for epoch in range(num_epoch):
    epoch_loss = 0
    for step,(x,y) in enumerate(train_loader): 
        model.train()
        
        h_pre,h_post,y_hat = model(Variable(x).to(device))
        
        # loss and backpropagation
        loss1 = criterion1(y_hat,Variable(y).to(device))
        loss2 = criterion2(y_hat,Variable(y).to(device))
        loss = loss1+loss2
        epoch_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('loss:{}'.format(loss.item()))
        
        if step % 1000 == 0:
            im_hat = y_hat[0,0,:,:].detach().cpu().numpy()
            im_pre = h_pre[0,0,:,:].detach().cpu().numpy()
            im_post = h_post[0,0,:,:].detach().cpu().numpy()
            im_y = y[0,0,:,:].detach().cpu().numpy()
            
            plt.figure(figsize=(16,16))
            plt.subplot(2,2,1),plt.axis('off'),plt.imshow(im_pre,cmap='gray')
            plt.subplot(2,2,2),plt.axis('off'),plt.imshow(im_post,cmap='gray')
            plt.subplot(2,2,3),plt.axis('off'),plt.imshow(im_hat,cmap='gray')
            plt.subplot(2,2,4),plt.axis('off'),plt.imshow(im_y,cmap='gray')
            plt.show()
            
    sum_loss.append(epoch_loss)
    scheduler.step()
t2 = time.time()

print('time:{} min'.format((t2-t1)/60))