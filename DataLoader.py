#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:17:16 2020

@author: hud4

This data loader is used for Conv2dLSTM training
The raw data is registered(both framewise and Bscanwise), type: float64
shape: [H,W,slc] 

1. crop the useful part
2. to type float32
3. x: [2r+1,H,W], y: [H,W]
4. save as paired (x,y) tuple in pickle file 

"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/20-summer/src/')
import util
import numpy as np
import os,pickle
import matplotlib.pyplot as plt

# re-arange the shape, type of the volume to [Frame,slc,H,W]
def ReShape(volume,FrameNum):
    v = np.transpose(volume,[2,1,0])
    dim = v.shape
    opt = np.zeros([FrameNum,int(dim[0]/FrameNum),dim[1],dim[2]],dtype=np.float32)
    for i in range(dim[0]):
        frame = i % FrameNum
        opt[frame,int(i/FrameNum),:,:] = v[i,:,:]
    return util.ImageRescale(opt,[0,255])

def Cropper(volume, axH, axW):
    v = volume[:,:,axH[0]:axH[1],axW[0]:axW[1]]
    return v

#%%

global FrameNum

FrameNum = 5
dataroot = '/sdb/Data/RgHuman/Fovea/'
namelist = []

for file in os.listdir(dataroot):
    if file.endswith('.nii'):
        namelist.append(file)
namelist.sort()

axH = [[200,700],[200,700],[100,600],[200,700],[0,500],[0,500]]
axW = [[40,540],[35,535],[40,540],[40,540],[40,540],[40,540]]


#%%
r = 2
data = ()

for i in range(len(namelist)):
    print('volume {} ...'.format(i+1))
    # reshape, crop and delete side slices
    V = ReShape(np.float32(util.nii_loader(dataroot+namelist[i])),FrameNum)
    V = Cropper(V,axH[i],axW[i])
    V = V[:,5:-5,:,:]
    
    # single frame and frame average
    Vx = V[0,:,:,:]
    Vy = np.mean(V,axis=0) 
    
    dim = Vx.shape
    for j in range(r,dim[0]-r):
        x = Vx[j-r:j+r+1,:,:]
        y = Vy[j,:,:]
        data = data+((x,y),)

with open('/home/hud4/Desktop/2020/Data/train.pickle','wb') as f:
    pickle.dump(data,f)

#plt.figure(figsize=(12,12))
#plt.imshow(V[0,100,:,:],cmap='gray')
#plt.axis('off')
#plt.show()

#%%
#util.nii_saver(V[0,10:-10,:,:],'/home/hud4/Desktop/','v.nii.gz')
