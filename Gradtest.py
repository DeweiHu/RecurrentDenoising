#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:24:16 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/2020/RNN/')
import Conv2dLSTM_v2
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
modelroot = '/home/hud4/Desktop/2020/Model/'

# Model
input_ch = 1
hidden_ch = [16,32,8,1]
kernel_size = (3,3)
num_layers = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Conv2dLSTM_v2.ConvLSTM(input_ch,hidden_ch,kernel_size,True)
model.load_state_dict(torch.load(modelroot+'LSTM_4layers.pt'))
model = model.to(device)

layers = []
for n, p in model.named_parameters():
    if(p.requires_grad) and ("bias" not in n):
        layers.append(n)

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

#%%
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

#%% loss
num_epoch = 5

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
    grad_mat = np.zeros([len(layers),len(train_loader)],dtype=np.float32)
    
    for step,(x,y) in enumerate(train_loader): 
        model.train()
        
        y_hat = model(Variable(x).to(device),hidden_ch)
        
        # loss and backpropagation
        loss1 = criterion1(y_hat,Variable(y).to(device))
        loss2 = criterion2(y_hat,Variable(y).to(device))
        loss = 10*loss1+3*loss2
        epoch_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # collect gradient magnitude
        ave_grads = []
        for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                ave_grads.append(p.grad.abs().mean().item())
        for i in range(len(layers)):
            grad_mat[i,step] = np.float32(ave_grads[i])
            
        # visualize the gradient flow
        if step % 200 == 0 and step != 0:
            print('loss:{}'.format(loss1.item()))
#            plot_grad_flow(model.named_parameters())
        
        if step == len(train_loader)-1:
            
            im_hat = util.ImageRescale(y_hat[0,0,:,:].detach().cpu().numpy(),[0,255])
            im_y = util.ImageRescale(y[0,0,:,:].detach().cpu().numpy(),[0,255])
            
            plt.figure(figsize=(16,8))
            plt.axis('off')
            plt.imshow(np.concatenate((im_hat,im_y),axis=1),cmap='gray')
            plt.show()
            # plot the magnitude of gradient
            Nparam, Niter = grad_mat.shape

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
            plt.show()
                
    sum_loss.append(epoch_loss)
    scheduler.step()
    
    del im_hat, im_y, grad_mat, ave_grads
    
t2 = time.time()

print('time:{} min'.format((t2-t1)/60))

#%%
name = 'LSTM_4layers.pt'
torch.save(model.state_dict(),modelroot+name)
