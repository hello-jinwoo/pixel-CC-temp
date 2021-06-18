import numpy as np
import torch
import torch.nn as nn

def get_abstract_illum_map(img, pool_size=8):
	pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
	return pool(img)