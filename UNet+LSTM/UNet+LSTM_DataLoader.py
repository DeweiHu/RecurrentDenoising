# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 23:38:48 2020

@author: hudew

Output data dimension and intensity range
X_seq: [nSeq,nBscan,H,W],[0,255]
Y_seq: [nSeq,nBscan,H,W],[0,255]
.pickle: ((X_seq, Y_seq),)

"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util
import MotionCorrection as MC

import pickle, time, os
import numpy as np
import matplotlib.pyplot as plt

def GetPair(Vx, Vy, pair, verbose):
    global n_seq
    nBscan, H, W = Vx.shape
    slc = np.random.randint(n_seq,nBscan-1,1)
    
    # for easy application, first few slices abandoned
    for i in range(n_seq,nBscan):
        x_ipt = Vx[i-n_seq:i+1,:,:]
        y_ipt = Vy[i-n_seq:i+1,:,:]
        x = np.zeros([n_seq,H,H],dtype=np.float32)
        y = np.zeros([n_seq,H,H],dtype=np.float32)
        
        im_fix = np.ascontiguousarray(np.float32(y_ipt[-1,:,:]))
        for j in range(n_seq):
            x_mov = np.ascontiguousarray(np.float32(x_ipt[j,:,:]))    
            y_mov = np.ascontiguousarray(np.float32(y_ipt[j,:,:]))
            x[j,:,:W] = MC.MotionCorrect(im_fix, x_mov)
            y[j,:,:W] = MC.MotionCorrect(im_fix, y_mov)
        
        if verbose == True and i == slc:
            im_x = np.concatenate((x[0,:,:],x[-1,:,:]),axis=1)
            im_y = np.concatenate((y[0,:,:],y[-1,:,:]),axis=1)
            plt.figure(figsize=(12,12))
            plt.axis('off')
            plt.title('slice {}'.format(slc),fontsize=15)
            plt.imshow(np.concatenate((im_x,im_y),axis=0),cmap='gray')
            plt.show()
            
        pair = pair + ((x,y),)
    return pair
            
if __name__ == '__main__':
    dataroot = 'E:\\HumanData\\'
    
    global n_seq
    n_seq = 3
    
    fovea_x = []
    fovea_y = []
    pair = ()
    
    for file in os.listdir(dataroot):
        if file.startswith('HN_Fovea') and file.endswith('.nii.gz'):
            fovea_x.append(file)
        elif file.startswith('LN_Fovea') and file.endswith('.nii.gz'):
            fovea_y.append(file)
    fovea_x.sort()
    fovea_y.sort()
    
    t1 = time.time()
    for i in range(len(fovea_x)):
        print('volume: {} pairing...'.format(fovea_x[i]))
        Vx = util.nii_loader(dataroot+fovea_x[i])
        Vy = util.nii_loader(dataroot+fovea_y[i])
        Vx = Vx[0,:,:,:]
        pair = GetPair(Vx,Vy,pair,True)
    t2 = time.time()
    print('train data paired. time: {} min'.format((t2-t1)/60))
    
    with open(dataroot+'UNet+LSTM_train.pickle','wb') as handle:
        pickle.dump(pair,handle)
        
    




