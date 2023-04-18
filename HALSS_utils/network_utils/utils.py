from model_arch import *
from augment import *
from utils import *
import os
import cv2
import tqdm
import hashlib
import requests

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

def dataloader():
    imgs = np.zeros((100,480,720,3)).astype(np.uint8)
    masks = np.zeros((100,480,720,3)).astype(np.uint8)
    j = 1
    for i in range(100):
        imgs[i] = cv2.imread("E:\\AirSim\\PythonClient\\multirotor\\custom_seg\\input\\images\\sub100\\images\\render" + str(j).zfill(4) + ".png")
        masks[i] = cv2.imread("E:\AirSim\PythonClient\multirotor\custom_seg\input\images\sub100\masks\ground" + str(j).zfill(4) + ".png")
        j += 1
    return imgs, masks

def dataloader_airsim():
    imgs = np.zeros((68,144,256,3)).astype(np.uint8)
    masks = np.zeros((68,144,256,3)).astype(np.uint8)
    j = 1
    for i in range(68):
        imgs[i] = cv2.imread("E:\\AirSim\\PythonClient\\multirotor\\custom_seg\\input\\airsim_drone\\" + str(j) + "_scene.png")
        masks[i] = cv2.imread("E:\\AirSim\\PythonClient\\multirotor\\custom_seg\\input\\airsim_drone\\" + str(j) + "_maskedSegmentation.png")
        j += 1
    return imgs, masks

def binarize(masks):
    index = masks.shape
    binary_masks = np.zeros((68,144,256)).astype(np.uint8)
    for i in range(index[0]):
        gray_mask = cv2.cvtColor(masks[i], cv2.COLOR_BGR2GRAY)
        _, binary_masks[i] = cv2.threshold(gray_mask, 0, 1, cv2.THRESH_OTSU)
    return binary_masks
        


#imgs, masks = dataloader()
#binary_masks = binarize(masks)
#cv2.imshow('1', imgs[0])
#cv2.imshow('2', masks[0])
#cv2.imshow('3', binary_masks[0])
#cv2.waitKey()
#cv2.destroyAllWindows()