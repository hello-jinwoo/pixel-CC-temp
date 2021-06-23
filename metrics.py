import numpy as np
import torch
import math
import cv2

def get_mae(img1, img2):
	'''
	IN: img1 & img2 ([B, 2, H, W])
	OUT: 
		- mae_per_batch ([1,])
		- mae_per_image ([B,])
		- mae_per_pixel ([B, H, W])
	'''
	g_tensor = torch.ones_like(img1)[:, 0:1, :, :]

	img1 = torch.cat([img1, g_tensor], dim=1) # [B, 3, H, W]
	img2 = torch.cat([img2, g_tensor], dim=1) # [B, 3, H, W]

	img1_reshaped = torch.reshape(img1, (img1.shape[0], 3, -1)) # [B, 3, H*W]
	img2_reshaped = torch.reshape(img2, (img2.shape[0], 3, -1)) # [B, 3, H*W]

	img1_norm = torch.norm(img1_reshaped, dim=1)[:, None, :] # [B, 1, H*W]
	img2_norm = torch.norm(img2_reshaped, dim=1)[:, None, :] # [B, 1, H*W]

	img1_normalized = img1_reshaped / (1e-6 + img1_norm) # [B, 3, H*W]
	img2_normalized = img2_reshaped / (1e-6 + img2_norm) # [B, 3, H*W]

	img_multiplication = img1_normalized * img2_normalized # [B, 3, H*W]
	cos_similarity = torch.clip(torch.sum(img_multiplication, dim=1), 0, 1) # [B, H*W]
	
	mae = torch.acos(cos_similarity) * 180 / np.pi # [B, H*W]

	mae_per_batch = torch.mean(mae).unsqueeze(dim=0) # [1,]
	mae_per_image = torch.mean(mae, dim=-1) # [B,]
	mae_per_pixel = torch.reshape(mae, (img1.shape[0], img1.shape[2], img1.shape[3])) # [B, H, W]

	return mae_per_batch, mae_per_image, mae_per_pixel

def get_psnr(pred, GT, white_level):
    """
    pred & GT   : (b,h,w,c) numpy array 3 channel RGB

    returns     : average PSNR of two images
    """  
    if white_level != None:
        pred = np.clip(pred, 0, white_level)
        GT = np.clip(GT, 0, white_level)

    return cv2.PSNR(pred, GT, white_level)