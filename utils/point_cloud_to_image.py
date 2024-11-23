from math import isnan
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.linalg
import numpy as np
import cv2
import os
import time
import shutil
import os
import torch
import math
import open3d as o3d
from scipy import stats

## Functions:

def numpy_array_to_tensor_gpu(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.cuda()
    return tensor

def surface_normal_from_interp_model(x_grid, y_grid, Z_interp, params):
    grad = np.gradient(Z_interp, y_grid, x_grid)
    dx = grad[1]
    dy = grad[0]
    normal = np.zeros((params.grid_res,params.grid_res,3))

    normal[:,:,0] = -dx
    normal[:,:,1] = -dy
    normal[:,:,2] = np.ones((params.grid_res,params.grid_res))
    normal_unit = np.zeros((params.grid_res,params.grid_res,3))
    normal_unit = normal/np.linalg.norm(normal, axis = 2).reshape(params.grid_res,params.grid_res,1)

    normal_unit_color = np.multiply((normal_unit+1), 255*.5)
    channel_red = normal_unit_color[:,:,1]
    channel_green = normal_unit_color[:,:,0]
    channel_blue = normal_unit_color[:,:,2]
    normal_image = np.flipud(np.dstack((channel_red, channel_green, channel_blue)).astype(np.uint8))
    return normal_image

# params 
# grid_res = 320

def collect_lidar(client):
    xyz_data = []
    t0_lidar = time.time()
    lidar_data = client.getLidarData(lidar_name = "LidarSensor2" ,vehicle_name = "Drone1")    # get general lidar data
    tf_lidar = time.time() - t0_lidar
    t0_lidarprocess = time.time()
    for i in range(int(len(lidar_data.point_cloud)/3)):   # xyz data to array
        xyz_data.append([lidar_data.point_cloud[i*3+1], lidar_data.point_cloud[i*3], -lidar_data.point_cloud[i*3+2]])
    data = np.array(xyz_data)
    tf_lidarprocess = time.time()  - t0_lidarprocess
    return data

def maximum_possible_points(pcd, x_cell_size, y_cell_size):
    x_bins = np.arange(pcd[:,0].min(), pcd[:,0].max(), x_cell_size)
    y_bins = np.arange(pcd[:,1].min(), pcd[:,1].max(), y_cell_size)
    return len(x_bins)*len(y_bins)*2