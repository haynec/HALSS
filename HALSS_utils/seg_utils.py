from pyparsing import col
from HALSS.HALSS_utils.network_utils.model_arch_dropout import *
from HALSS.HALSS_utils.network_utils.augment import *
#from point_cloud_to_image import pc_surf_normal
#from traj_utils import *
import os
import tqdm
import hashlib
import requests
import threading

#from numba import jit

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

# Debugger
from pdb import set_trace as debug

flag_use_pc_surf_norm = True
flag_use_mc_dropout = True

def getRotMatrix_uv2worldNED(client):
        cam_info = client.simGetCameraInfo(3)
        kin = client.simGetGroundTruthKinematics()
        orientation = kin.orientation
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                        [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                        [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
        return rotation_matrix, kin, cam_info

def normalize(img):
  X = img.copy()
  xmin = np.amin(X)
  xmax = np.amax(X)
  X = (X - xmin) / (xmax - xmin)
  return X.astype(np.float32)

def torch_tensor_vis(tensor, name):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = 255*tensor.squeeze()
    tensor = tensor.astype(np.uint8)
    cv2.imshow(name, tensor)
    return

def numpy_array_vis(array, name):
    array = array.squeeze()
    cv2.imshow(name, array)
    cv2.waitKey(1)
    return

def torch_to_numpy(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = tensor.squeeze()
    return tensor

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")

def getCombinedMask(surfNorm, seg):
 
    surfaceNormal_vector_form = surfNorm.copy()

    # conversion from surface normal to pointing vector: color value / 255 * 2.0 - 1.0
    w,h,channels = surfNorm.shape
    for i in range(channels):
        for j in range(w):
            for k in range(h):
                color_value = surfNorm[j,k,i]
                temp = int(color_value / 255 * 2.0 - 1.0)
                surfaceNormal_vector_form[j,k,i] = temp*255

    # let's get rid of the x and y components, we only care about z component
    surfaceNormal_vector_form[:,:,1] = 0
    surfaceNormal_vector_form[:,:,2] = 0

    # filter out small noise
    filtered = cv2.bilateralFilter(surfaceNormal_vector_form,30,150,150)
  
    # blur to get get smoother shapes without gaps
    blurred = cv2.GaussianBlur(filtered,(5,5),5)
    
    # blurred = cv2.GaussianBlur(surfaceNormal_vector_form,(5,5),5)

    # convert to gray for binairzation and thresholding
    gray_mask = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    b, surfaceNormal_mask = cv2.threshold(gray_mask,0,256,cv2.THRESH_BINARY)

    gray_mask = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    b, segmentation_mask = cv2.threshold(gray_mask,0,256,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_and(surfaceNormal_mask, segmentation_mask)
    return combined_mask, surfaceNormal_mask, segmentation_mask

def topNCircles(distance_map_original,N):
    distance_map = distance_map_original.copy()
    centers = []
    radii = []
    for n in range(N):
        center = np.where(distance_map == np.max(distance_map))
        xc = center[1][0]
        yc = center[0][0]
        radius = np.int64(np.max(distance_map))
        
        distance_map = cv2.circle(distance_map, (xc,yc), radius, (0,0,0), -1)
        centers.append(center)
        radii.append(radius)
    return centers, radii

def plotCircles(centers, radii, image, color = None, thickness = 1):
    image_clone = image.copy()
    color_fill = (230,230,255)
    color = (0,0,255)
    w = image_clone.shape[0]
    h = image_clone.shape[1]
    if len(image_clone.shape) == 2:
        color_img = np.zeros((w,h,3)).astype(np.uint8)
        color_img[:,:,0] = image_clone
        color_img[:,:,1] = image_clone
        color_img[:,:,2] = image_clone
    else:
        color_img = image_clone
    
    if type(centers[0]) is int: # Captures the case of only one circle
        idx = 1
    else:
        idx = len(centers[0])
    
    if type(radii) is int: # Captures the case of only one circle
        radii = [radii]
    else:
        radii = radii
        
    for i in range(idx):
        radius = int(radii[i])
        if type(centers[0]) is int:
            xc = int(centers[0]+2)
            yc = int(centers[1]+2)
        else:
            xc = int(centers[0][i]+2)
            yc = int(centers[1][i]+2)
        color_img = cv2.circle(color_img, (yc,xc), radius, color_fill, cv2.FILLED)
        color_img = cv2.circle(color_img, (yc,xc), radius, color, thickness)
        color_img = cv2.circle(color_img, (yc,xc), 1, color, -1)
    return color_img

def landing_selection(mask, num_circles = 50):
    data = prep_safety_mask(mask)
    
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)
    dist_on_skel = distance * skel


    center_coords, radii = topNCircles(distance, num_circles)
    u_vec = np.zeros(len(center_coords))
    v_vec = np.zeros(len(center_coords))
    for i in range(len(center_coords)):
        u_vec[i] = center_coords[i][0][0]
        v_vec[i] = center_coords[i][1][0]

    circles = plotCircles((u_vec, v_vec), radii, data)
    return radii, center_coords, dist_on_skel, circles, distance

def remove_pixels_around_boarder(mask, border_size = 2):
    mask = mask.copy()
    mask[0:border_size,:] = 0
    mask[mask.shape[0]-border_size:,:] = 0
    mask[:,0:border_size] = 0
    mask[:,mask.shape[1]-border_size:] = 0
    return mask

            
def prep_safety_mask(mask):
    topBorderWidth = 2
    bottomBorderWidth = 2
    leftBorderWidth =2
    rightBorderWidth = 2
    data = cv2.copyMakeBorder(
                    mask, 
                    topBorderWidth, 
                    bottomBorderWidth, 
                    leftBorderWidth, 
                    rightBorderWidth, 
                    cv2.BORDER_CONSTANT, 
                    value=0
                )
    data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return data

def prep_surf_norm(mask):
    topBorderWidth = 2
    bottomBorderWidth = 2
    leftBorderWidth =2
    rightBorderWidth = 2
    data = cv2.copyMakeBorder(
                    mask, 
                    topBorderWidth, 
                    bottomBorderWidth, 
                    leftBorderWidth, 
                    rightBorderWidth, 
                    cv2.BORDER_CONSTANT, 
                    value=0
                )
    return data