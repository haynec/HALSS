# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import os

from custom_seg.model_arch_dropout import *
from custom_seg.augment import *
from seg_utils import *
import tqdm
import hashlib
import requests
import random

import numpy as np
import matplotlib.pyplot as plt

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import find_objects, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from scipy.optimize import linear_sum_assignment
from skimage.morphology import medial_axis
import scipy.ndimage

import time
import sys


# #####################
# Network Setup
# #####################
kernel_size = 3
nbase = [3, 32, 64, 128, 256]  # number of channels per layer
nout = 1  # number of outputs

#net = Unet(nbase, nout, kernel_size)
#print(net)
# put on GPU here if you have it
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(torch.cuda.get_device_name(0))

# compute results on test images
# (note for unet to run correctly we need to pad images to be divisible by 2**(number of layers))
#net.load_model(f"E:\\AirSim\\PythonClient\\multirotor\\unet_epoch1000.pth")
#net.eval()


png = cv2.imread("C:\\Users\\chris\\Documents\\HA_PDG\AirSim\Release\\custom_seg\\airsim_drone_608_200\\1_surfaceNormal.png")
cv2.imshow('Network Input', png)

net = Unet_drop(nbase, nout, kernel_size)
#net.load_model(f"E:\\AirSim\\PythonClient\\multirotor\\unet_epoch1000.pth")
net.load_model(f"AirSim\\Release\\unet_epoch1000.pth")
net.eval()
net.to(device);  # remove semi-colon to see net structure

# Run the segmentation network
current = normalize(cv2.resize(png, (320, 320)).transpose(2,0,1))
img_torch = torch.from_numpy(current).to(device).unsqueeze(0)  # also need to add a first dimension
time.sleep(3)

i = 0
mc_num = 30



t0_new = time.time()
img_torch_batch = torch.repeat_interleave(img_torch, 30, dim = 0)
new_out = net(img_torch_batch)
tf_new = time.time() - t0_new
print("Inference time for {} MC samples using repeat_interleave: {}".format(mc_num, tf_new))



# t0 = time.time()
# out = torch.zeros(mc_num,1,320,320)
# for idx in range(mc_num):
#   out[i] = net(img_torch)
#   i += 1
# tf = time.time() - t0
# print("Inference time for {} MC samples using for loop: {}".format(mc_num, tf))

png = cv2.imread("C:\\Users\\chris\\Documents\\HA_PDG\AirSim\Release\\custom_seg\\airsim_drone_608_200\\5_surfaceNormal.png")
current = normalize(cv2.resize(png, (320, 320)).transpose(2,0,1))
img_torch = torch.from_numpy(current).to(device).unsqueeze(0)  # also need to add a first dimension




t0_new_2 = time.time()
img_torch_batch = torch.repeat_interleave(img_torch, 30, dim = 0)
new_out = net(img_torch_batch)
tf_new_2 = time.time() - t0_new_2
print("Inference time for {} MC samples using repeat_interleave #2: {}".format(mc_num, tf_new_2))

t0_new_3 = time.time()
img_torch_batch = torch.repeat_interleave(img_torch, 30, dim = 0)
new_out = net(img_torch_batch)
tf_new_3 = time.time() - t0_new_3
print("Inference time for {} MC samples using repeat_interleave #3: {}".format(mc_num, tf_new_3))




mean_out = torch.mean(new_out,0)
var_out = torch.var(new_out,0)
var_out = normalize(torch_to_numpy(var_out))

torch_tensor_vis(mean_out, "Mean Output")
numpy_array_vis(var_out, "Variance Output")

cv2.waitKey(0)
cv2.destroyAllWindows()