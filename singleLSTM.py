#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 12:44:00 2020

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
hidden_ch = 1
kernel_size = (3,3)
num_layers = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Conv2dLSTM.ConvLSTM(input_ch,hidden_ch,kernel_size,num_layers,True,True,False)
model.load_state_dict(torch.load('/home/hud4/Desktop/2020/Model/'+'single_LSTM.pt'))
model = model.to(device)

#check parameter number
#for i in range(len(list(model.parameters()))):
#    print(list(model.parameters())[i].size())

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.figure(figsize=(10,8))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
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
#
#im = y[0,0,:,:].numpy()
#print(x.size())
#print(y.size())
    
#%% loss
num_epoch = 1

criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
r = 2

sum_loss = []


t1 = time.time()

for epoch in range(num_epoch):
    epoch_loss = 0
    grad_mat = np.zeros([8,len(train_loader)],dtype=np.float64)
    for step,(x,y) in enumerate(train_loader): 
        model.train()
        
        _,last_states = model(Variable(x).to(device))
        y_hat = last_states[0][0]
#        h_pre,h_post,y_hat = model(Variable(x).to(device))
        
        # loss and backpropagation
        loss1 = criterion1(y_hat,Variable(y).to(device))
        loss2 = criterion2(y_hat,Variable(y).to(device))
        loss = 10*loss1+3*loss2
        epoch_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # visualize the gradient flow
        if step % 200 == 0 and step != 0:
            print('loss:{}'.format(loss1.item()))
            plot_grad_flow(model.named_parameters())
        
        if step % 1000 == 0:
            im_hat = y_hat[0,0,:,:].detach().cpu().numpy()
            im_y = y[0,0,:,:].detach().cpu().numpy()
#            im_pre = h_pre[0,0,:,:].detach().cpu().numpy()
#            im_post = h_post[0,0,:,:].detach().cpu().numpy()
            
            plt.figure(figsize=(16,8))
            plt.subplot(1,2,1),plt.axis('off'),plt.imshow(im_hat,cmap='gray')
            plt.subplot(1,2,2),plt.axis('off'),plt.imshow(im_y,cmap='gray')
#            plt.subplot(2,2,3),plt.axis('off'),plt.imshow(im_pre,cmap='gray')
#            plt.subplot(2,2,4),plt.axis('off'),plt.imshow(im_post,cmap='gray')
            plt.show()
    
        layers = []
        ave_grads = []
        for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
        for i in range(8):
            grad_mat[i,step] = ave_grads[i]
                
    sum_loss.append(epoch_loss)
    scheduler.step()
    
t2 = time.time()

print('time:{} min'.format((t2-t1)/60))

#%%
#stat = []
#for name, param in model.named_parameters():
#    pair = (name,param.grad)
#    stat.append(pair)

name = 'single_LSTM.pt'
modelroot = '/home/hud4/Desktop/2020/Model/'
torch.save(model.state_dict(),modelroot+name)

#%%
a = grad_mat[:,:100]
Nparam, Niter = a.shape

plt.figure(figsize=(10,8))
for i in range(Niter):
    plt.plot(grad_mat[:,i], alpha=0.3, color="b")
    plt.hlines(0, 0, Nparam+1, linewidth=1, color="k" )
    plt.xticks(range(0,Nparam, 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
