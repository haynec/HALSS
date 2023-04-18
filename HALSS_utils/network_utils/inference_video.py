# Imports
from model_arch import *
from augment import *
import os
import cv2
import tqdm
import hashlib
import requests
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from numba import jit

from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import find_objects, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from scipy.optimize import linear_sum_assignment


def normalize99(img):
  """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
  X = img.copy()
  x01 = np.percentile(X, 1)
  x99 = np.percentile(X, 99)
  X = (X - x01) / (x99 - x01)
  return X.astype(np.float32)



kernel_size = 3
nbase = [2, 32, 64, 128, 256]  # number of channels per layer
nout = 2  # number of outputs

net = Unet(nbase, nout, kernel_size)
print(net)
# put on GPU here if you have it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(torch.cuda.get_device_name(0))
net.to(device);  # remove semi-colon to see net structure

# compute results on test images
# (note for unet to run correctly we need to pad images to be divisible by 2**(number of layers))
net.load_model(f"unet_epoch50.pth")
net.eval()
cap = cv2.VideoCapture("E:\AirSim\PythonClient\multirotor\custom_seg\input\video1.mp4")
width = int(cap.get(3))
height = int(cap.get(4))
while True:
  ret, frame = cap.read()
  if ret == True:
    start_time = time.time()
    with torch.no_grad():
      #img_padded, slices = pad_image_ND(frame, 8)
      current = normalize99(frame.reshape((3, 360, 640)))
      img_torch = torch.from_numpy(current[0:2,:,:]).to(device).unsqueeze(0)  # also need to add a first dimension
      out = net(img_torch)
      labels = out[0].detach().cpu()
      np_labels = labels.numpy()
      cv2.imshow('Prediction 1', np_labels[0,:,:])
      cv2.imshow('Prediction 2', np_labels[1,:,:])
      cv2.waitKey(1)
