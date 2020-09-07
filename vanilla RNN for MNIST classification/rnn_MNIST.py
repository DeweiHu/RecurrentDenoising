# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:36:58 2020

@author: hudew
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as Data
from torch.autograd import Variable

#%% load data
root = 'C:\\Users\\hudew\\OneDrive\\桌面\\RNN\\data\\'

'''
train_dataset.train_data: [60000,28,28]
train_dataset.train_label: [60000]
'''
train_dataset = dataset.MNIST(root=root,
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

'''
test_dataset.test_data: [10000,28,28]
test_dataset.test_label: [10000]
'''
test_dataset = dataset.MNIST(root=root,
                             train=False,
                             transform=transforms.ToTensor())

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               shuffle=False)

#%% Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # dim of h
        self.hidden_dim = hidden_dim
        # number of layers
        self.layer_dim = layer_dim
        # [batchsize, seq_dim, feature_dim]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        # initial hidden state & cell state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        # backpropagation through time (BPTT)
        out,(hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
        out = self.fc(out[:,-1,:])
        return out
    
#%% Hyperparameters
# x:[batchsize,seq_dim,input_dim]
input_dim = 28
seq_dim = 28

# h:[layer_dim,batchsize,hidden_dim]
hidden_dim = 100
layer_dim = 1

# y:[batchsize,output_dim]
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-4)

#%% Training
for epoch in range(num_epochs):
    for step,(x,y) in enumerate(train_loader):
        # x:[batchsize,seq_dim,input_dim]
        x = x.view(-1,seq_dim,input_dim).requires_grad_()
        y_hat = model(x)

        optimizer.zero_grad()
        loss = criterion(y_hat,y)
        loss.backward()
        optimizer.step()
        
        # testing
        if step % 500 == 0:
            correct = 0
            total = 0
            for x,y in test_loader:
                x = x.view(-1,seq_dim,input_dim)
                y_hat = model(x)
                
                _,predicted = torch.max(y_hat.data,1)
                
                total += y.size(0)
                correct += (predicted == y).sum()
            accuracy = 100*correct/total
            
            print('Iteration: {}, Loss: {}, Accuracy: {}'.format(step,
                  loss.item(), accuracy))
