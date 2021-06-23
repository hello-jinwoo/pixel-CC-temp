import numpy as np
import torch
import torch.nn as nn
import cv2

def get_abstract_illum_map(img, pool_size=8):
	pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
	return pool(img)

def apply_wb(I_patch, pred, output_mode):
    pred_patch = torch.zeros_like(I_patch) # b,h,w,c
    
    if output_mode == 'uv':
        pred_patch[:,1,:,:] = I_patch[:,1,:,:]
        pred_patch[:,0,:,:] = I_patch[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_patch[:,2,:,:] = I_patch[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)

    elif output_mode == 'illumination':
        pred_patch[:,1,:,:] = I_patch[:,1,:,:]
        pred_patch[:,0,:,:] = I_patch[:,0,:,:] * (1 / (pred[:,0,:,:]+1e-8))    # R = R * (1/R_i)
        pred_patch[:,2,:,:] = I_patch[:,2,:,:] * (1 / (pred[:,1,:,:]+1e-8))    # B = B * (1/B_i)

    return pred_patch