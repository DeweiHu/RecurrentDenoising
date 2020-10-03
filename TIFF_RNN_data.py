# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:41:25 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util
import MotionCorrection as MC

import os, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

dataroot = 'E:\\HumanData\\'
HN_list = []
LN_list = []

for file in os.listdir(dataroot):
    if file.startswith('HN') and file.endswith('2.nii.gz'):
        HN_list.append(file)
    elif file.startswith('LN') and file.endswith('2.nii.gz'):
        LN_list.append(file)
HN_list.sort()
LN_list.sort()

def BscanRegist(x):
    nBscan,H,W = x.shape
    opt = np.zeros([nBscan,H,H],dtype=np.float32)
    im_fix = np.ascontiguousarray(np.float32(x[-1,:,:]))
    
    for i in range(nBscan):
        im_mov = np.ascontiguousarray(np.float32(x[i,:,:]))
        opt[i,:,:W] = MC.MotionCorrect(im_fix, im_mov)
    return opt

#%%
n_seq = 5
train_data = ()

t1 = time.time()
for i in range(len(HN_list)):
    print('volume {} registering...'.format(i+1))
    # [nBscan,H,W]
    vol_x = util.nii_loader(dataroot+HN_list[i])
    vol_y = util.nii_loader(dataroot+LN_list[i])
    nBscan, H, W = vol_x.shape
    
    for j in range(n_seq,nBscan):
        x = BscanRegist(vol_x[j-5:j,:,:])
        y = vol_y[j,:,:]
        train_data = train_data+((x,y),)
        
        if j == 300:
            r1 = np.concatenate((x[0,:,:],x[1,:,:],x[2,:,:]),axis=1)
            r2 = np.concatenate((x[3,:,:],x[4,:,:],y),axis=1)
            plt.figure(figsize=(12,8))
            plt.axis('off'),plt.title('Regist Result',fontsize=15)
            plt.imshow(np.concatenate((r1,r2),axis=0),cmap='gray')
            plt.show()

with open('TIFF_RNN_data.pickle','wb') as handle:
    pickle.dump(train_data,handle)
t2 = time.time()

print('done. time: {} min'.format((t2-t1)/60))

