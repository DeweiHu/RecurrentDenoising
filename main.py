# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:30:14 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\RNN\\')
from Conv2dLSTM import ConvLSTM
import util

import time,os
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

# dataloader
dataroot = 'E:\\HumanData\\'
trainlist = []

for file in os.listdir(dataroot):
    if file.endswith("_2.nii"):
        trainlist.append(file)

#%%
#plt.figure(figsize=(12,12))
#plt.imshow(data[200:700,33:533,1000],cmap='gray')
#plt.imshow()         

#%% 
global NumFrame
NumFrame = 5

batch_size = 1

# [(x,y)], x:[batch,seq_dim,channel,H,W]

class MyDataset(Data.Dataset):

    def ToTensor(self, x, y):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=1)  # add channel dimension
        y_tensor = transforms.functional.to_tensor(y)
        return x_tensor, y_tensor    
        
    def __init__(self, dir):
       self.data = util.nii_loader(dir)
       self.data = util.ImageRescale(np.float32(np.transpose(self.data,[2,1,0])),[0,255])
       self.data = self.data[:100,200:700,33:533]
       
       self.pair = ()
       for i in range(100):
           if i % NumFrame == 0:
               x = self.data[i:i+NumFrame,:,:]
               y = np.mean(x,axis=0)
               self.pair = self.pair + ((x,y),)
        
    def __len__(self):
        return len(self.pair)

    def __getitem__(self,idx):
        (x, y) = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(x,y)
        return x_tensor, y_tensor
    
train_loader = Data.DataLoader(dataset=MyDataset(dataroot+trainlist[0]), 
                               batch_size=batch_size, shuffle=True)


#%%
#for step,(x,y) in enumerate(train_loader):
#    pass
#print(x.size())
#print(y.size())

#%% Model
# initialization 
model = ConvLSTM(input_dim = 1, hidden_dim = 1,
                 kernel_size = (3,3), num_layers = 3,
                 batch_first = True, bias = True,
                 return_all_layers = False).cuda()

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
#%% loss
num_epoch = 10
criterion = nn.MSELoss().cuda()
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-4)

for epoch in range(num_epoch):
    for step,(x,y) in enumerate(train_loader): 
        _, last_states = model(Variable(x).cuda(),None)
        y_hat = last_states[0][0]
        
        # loss and backpropagation
        loss = criterion(y_hat,Variable(y).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(step)

    
    
    


