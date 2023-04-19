# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
from distutils.command.build import build
from multiprocessing.spawn import old_main_modules


import sys
import os
sys.path.append(os .getcwd())


from HALSS.HALSS_utils.point_cloud_to_image import maximum_possible_points, surface_normal_from_interp_model
from HALSS.HALSS_utils.network_utils.model_arch import *
from HALSS.HALSS_utils.network_utils.augment import *
from HALSS.HALSS_utils.seg_utils import *
#from AirSim.utils.airsim_traj_utils import nparray2vector3rlist

import numpy as np

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from scipy import interpolate
import scipy.ndimage

import time

def convert_grey_to_color(safety_map):
  color_safety_map = np.zeros((3,320,320))
  color_safety_map[0] = safety_map
  color_safety_map[1] = safety_map
  color_safety_map[2] = safety_map
  color_safety_map = color_safety_map.transpose(1,2,0)
  return color_safety_map

def invert(png):
  # Invert the image
  inverted = 255 - png
  return inverted

def score_landings(halss_data):
  # Scores each potential landing site with the following criteria:
  # 1. How large is the landing site by radius
  # 2. How close is the landing site in the XY NED frame to quadrotor, DEPRECIATED
  # 3. How many points from the LiDAR Pointcloud are within the landing site
  # 4. How large is the 99th percentile of the uncertainty map in the landing site
  # 5. How near a landing site is to other landing sites

  # Useful conversions
  #center_coords_ned = vector3r_list2np(center_coords_ned)[:,:2]

  # 1. How large is the landing site
  radius = np.array(halss_data.radii_uv)
  size_scores = radius

  # 2. How close is the landing site in the NED frame to quadrotor
  #drone_pos = vector3r2np(client.simGetGroundTruthKinematics().position)
  #drone_scores = np.linalg.norm(center_coords_ned - drone_pos, axis = 1)
  drone_scores = np.ones_like(radius)

  # 3. How many points from the LiDAR are within the landing site
  halss_data.scale_uv_2_world()
  sf_x = halss_data.sf_x
  radius_ned = sf_x * radius
  density_score = np.zeros(len(radius))
  for idx in range(len(radius)):
    dist = (halss_data.pcd_full[:,0] - halss_data.center_coords_ned[idx][0])**2 + (halss_data.pcd_full[:,1] - (-halss_data.center_coords_ned[idx][1]))**2
    within = dist < radius_ned[idx]**2
    density_score[idx] = np.sum(within)/(np.pi*radius_ned[idx]**2)
  density_score = density_score/density_score.max()

  # 4. How large is the 99th percentile of the uncertainty map in the landing site
  if halss_data.variance_map is None:
    uncertainty_score = np.ones_like(radius)
  else:
    uncertainty_flipped = np.flipud(halss_data.variance_map)
    uncertainty_flipped = np.ascontiguousarray(uncertainty_flipped, np.float32)
    uncertainty_score = np.zeros(len(radius))
    bins = 100
    percentile = 99
    for idx in range(len(radius)):
      unc_flip_cp = np.copy(uncertainty_flipped)
      mask = np.zeros_like(unc_flip_cp, dtype = "uint8")
      cv2.circle(mask, (int(halss_data.center_coords_uv[idx][1]), int(halss_data.center_coords_uv[idx][0])), int(radius[0]), 255, -1)
      hist = cv2.calcHist([unc_flip_cp], [0], mask, [bins], [0.0, 1.001])
      sum = np.sum(hist)
      count = 0
      for hist_idx in range(len(hist)):
        count += hist[hist_idx]
        if count > sum*percentile:
          break
      uncertainty_score[idx] = float(hist_idx/bins)
    uncertainty_score = uncertainty_score/uncertainty_score.max()
    
  # 5. How near a landing site is to other landing sites
  prox_map = scipy.spatial.distance.cdist(halss_data.center_coords_ned, halss_data.center_coords_ned, metric='euclidean')
  prox_score = np.sum(prox_map, axis = 0)
  prox_score = 1/(prox_score/prox_score.max())
  prox_score = prox_score/prox_score.max()

  return np.vstack((size_scores, drone_scores, density_score, uncertainty_score, prox_score))



def scale_image(img, pcd_combined_array):
  # Takes in Image and min_x max_x min_y max_y of point cloud and scales the image proportionally
  resize_fac = ((pcd_combined_array[:,1].max()-pcd_combined_array[:,1].min())/(pcd_combined_array[:,0].max()-pcd_combined_array[:,0].min()))
  img = cv2.resize(img, (img.shape[0], int(img.shape[1]*resize_fac)))
  return img

def multi_fine_landing_site_selection(thread_idx, halss_global, params, flags, buffer = 1.1):
  #new_radii_ned = np.zeros_like(halss_global.radii_ned)
  new_center_coords_ned = np.zeros_like(halss_global.center_coords_ned[0])

  #for idx, radius_ned in enumerate(halss_global.radii_ned):
  halss_local = halss_data_packet()
  halss_local.num_circles = 1
  halss_local.x_cell_size = params.x_cell_size_fine
  halss_local.y_cell_size = params.y_cell_size_fine
  radius_ned = halss_global.radii_ned[thread_idx]

  dist_squared = (halss_global.pcd_full[:,0] - halss_global.center_coords_ned[thread_idx][0])**2 + (halss_global.pcd_full[:,1] - (-halss_global.center_coords_ned[thread_idx][1]))**2
  within = dist_squared < (buffer*radius_ned)**2
  halss_local.pcd_full = halss_global.pcd_full[within]
  halss_local.pcd_new = halss_local.pcd_full

  halss_local.downsample_pointcloud_dict()
  halss_local.downsample_pointcloud()
  
  if len(halss_local.pcd_culled) > 10:
    if flags.flag_debug:
      print("--> [[FINE SELECTION] DEBUG: Number of points in local pointcloud for Site " + str(thread_idx) + ": " + str(len(halss_local.pcd_culled)) + " out of " + str(maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size)) + " possible points]")
    if len(halss_local.pcd_culled) > maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size):
      print("--> [[FINE SELECTION] Warning!: Chris lost his mind downsampling small pointclouds]")
    halss_local.pc_to_surf_normal(params)
  else:
    print("--> [[FINE SELECTION] Warning!: There are probably not enough points in the landing site to generate a surface normal. Setting Radius to 0]")
    halss_local.radii_ned = 0.
    return halss_global

  halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (320, 320))

  halss_local.surf_norm_thresh(params)

  halss_local.sn_x_min = 0
  halss_local.sn_x_max = halss_local.safety_map.shape[1]
  halss_local.sn_y_min = 0
  halss_local.sn_y_max = halss_local.safety_map.shape[0]
  halss_local.find_NED_orgin_uv()

  halss_local.landing_selection(flags)
  
  if flags.flag_save_pointclouds:
    np.save("Fine Landing Site Pointcloud " + str(thread_idx) + ".npy", halss_local.pcd_culled)
  if flags.flag_show_single_safety_fine:
    cv2.imshow("Fine Landing Site Safety Map with Circle" + str(thread_idx), halss_local.safety_map_circles)
    cv2.imshow("Fine Landing Site Surface Normal " + str(thread_idx), halss_local.surf_norm)
    cv2.imshow("Fine Angle Map " + str(thread_idx), scale_theta(halss_local.angle_map))
  if flags.flag_save_images:
    cv2.imwrite(params.media_save_path + "Fine Landing Site Safety Map with Circle " + str(thread_idx) + ".png", halss_local.safety_map_circles)
    cv2.imwrite(params.media_save_path + "Fine Landing Site Safety Map " + str(thread_idx) + ".png", halss_local.safety_map)
    cv2.imwrite(params.media_save_path + "Fine Landing Site Surface Normal " + str(thread_idx) + ".png", halss_local.surf_norm)
    cv2.imwrite(params.media_save_path + "Fine Angle Map " + str(thread_idx) + ".png", scale_theta(halss_local.angle_map))
  halss_local.radii_ned = halss_local.sf_x*np.array((halss_local.radii_uv)).astype("float") # Convert new radii to numpy array
  new_center_coords_ned = np.squeeze(halss_local.center_coords_ned)
  halss_global.radii_ned[thread_idx] = halss_local.radii_ned  
  halss_global.center_coords_ned[thread_idx] = new_center_coords_ned.T
  return halss_global

def fine_landing_site_selection(halss_global, params, flags, buffer = 1.1):
  new_radii_ned = np.zeros_like(halss_global.radii_ned)
  new_center_coords_ned = np.zeros_like(halss_global.center_coords_ned.T)

  for idx, radius_ned in enumerate(halss_global.radii_ned):
    halss_local = halss_data_packet()
    halss_local.num_circles = 1
    halss_local.x_cell_size = params.x_cell_size_fine
    halss_local.y_cell_size = params.y_cell_size_fine

    dist_squared = (halss_global.pcd_full[:,0] - halss_global.center_coords_ned[idx][0])**2 + (halss_global.pcd_full[:,1] - (-halss_global.center_coords_ned[idx][1]))**2
    within = dist_squared < (buffer*radius_ned)**2
    halss_local.pcd_full = halss_global.pcd_full[within]
    halss_local.pcd_new = halss_local.pcd_full

    halss_local.downsample_pointcloud_dict()
    halss_local.downsample_pointcloud()
    
    if len(halss_local.pcd_culled) > 10:
      if flags.flag_debug:
        print("--> [[FINE SELECTION] DEBUG: Number of points in local pointcloud for Site " + str(idx) + ": " + str(len(halss_local.pcd_culled)) + " out of " + str(maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size)) + " possible points]")
      if len(halss_local.pcd_culled) > maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size):
        print("--> [[FINE SELECTION] Warning!: Chris lost his mind downsampling small pointclouds]")
      halss_local.pc_to_surf_normal(params)
    else:
      print("--> [[FINE SELECTION] Warning!: There are probably not enough points in the landing site to generate a surface normal. Setting Radius to 0]")
      halss_local.radii_ned = 0.
      continue

    halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (320, 320))

    halss_local.surf_norm_thresh(params)

    halss_local.sn_x_min = 0
    halss_local.sn_x_max = halss_local.safety_map.shape[1]
    halss_local.sn_y_min = 0
    halss_local.sn_y_max = halss_local.safety_map.shape[0]
    halss_local.find_NED_orgin_uv()

    halss_local.landing_selection(flags)
    
    if flags.flag_save_pointclouds:
      np.save("Fine Landing Site Pointcloud " + str(idx) + ".npy", halss_local.pcd_culled)
    if flags.flag_show_single_safety_fine:
      cv2.imshow("Fine Landing Site Safety Map with Circle" + str(idx), halss_local.safety_map_circles)
      cv2.imshow("Fine Landing Site Surface Normal " + str(idx), halss_local.surf_norm)
      cv2.imshow("Fine Angle Map " + str(idx), scale_theta(halss_local.angle_map))
    if flags.flag_save_images:
      cv2.imwrite(params.media_save_path + "Fine Landing Site Safety Map with Circle " + str(idx) + ".png", halss_local.safety_map_circles)
      cv2.imwrite(params.media_save_path + "Fine Landing Site Safety Map " + str(idx) + ".png", halss_local.safety_map)
      cv2.imwrite(params.media_save_path + "Fine Landing Site Surface Normal " + str(idx) + ".png", halss_local.surf_norm)
      cv2.imwrite(params.media_save_path + "Fine Angle Map " + str(idx) + ".png", scale_theta(halss_local.angle_map))
    halss_local.radii_ned = halss_local.sf_x*np.array((halss_local.radii_uv)).astype("float") # Convert new radii to numpy array
    new_radii_ned[idx] = halss_local.radii_ned
    new_center_coords_ned[:,idx] = np.squeeze(halss_local.center_coords_ned)
  halss_global.radii_ned = new_radii_ned  
  halss_global.center_coords_ned = new_center_coords_ned.T
  return halss_global

def scale_theta(img):
  img = img.astype(np.float32)
  img = img - np.min(img)
  img = img / np.max(img)
  img = img * 255
  img = img.astype(np.uint8)
  img = 255-img
  return img


def binarize(safety_map):
  safety_map[safety_map == 255] = 255
  safety_map[safety_map < 255] = 0
  return safety_map


def update_landing_site_radii(halss_global, params, flags, buffer = 2):
  t0_update_landing_site_func = time.time()
  new_radii_ned = np.zeros_like(halss_global.radii_ned)
  radii_local = int(160*(1/buffer))

  for idx, radius_ned in enumerate(halss_global.radii_ned):
    halss_local = halss_data_packet()
    halss_local.x_cell_size = params.x_cell_size_fine
    halss_local.y_cell_size = params.y_cell_size_fine
    
    dist_squared = (halss_global.pcd_full[:,0] - halss_global.center_coords_ned[idx][0])**2 + (halss_global.pcd_full[:,1] - (-halss_global.center_coords_ned[idx][1]))**2
    within = dist_squared < (buffer*radius_ned)**2
    halss_local.pcd_full = halss_global.pcd_full[within]
    halss_local.pcd_new = halss_local.pcd_full

    halss_local.downsample_pointcloud_dict()
    halss_local.downsample_pointcloud()
    
    if len(halss_local.pcd_culled) > 10:
      if flags.flag_debug:
        print("--> [[RADII UPDATE] DEBUG: Number of points in local pointcloud for Site " + str(idx) + ": " + str(len(halss_local.pcd_culled)) + " out of " + str(maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size)) + " possible points]")
      if len(halss_local.pcd_culled) > maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size):
        print("--> [[RADII UPDATE] Warning!: Chris can't downsample small pointclouds]")
      halss_local.pc_to_surf_normal(params)
    else:
      print("--> [[RADII UPDATE] Warning!: There are probably not enough points in the landing site to generate a surface normal. Setting Radius to 0]")
      halss_global.radii_ned[idx] = new_radii_ned = 0.
      return halss_global

    if halss_local.surf_norm.max == 0:
      print("--> [[RADII UPDATE] Warning!: Surface normal is all zeros. Setting Radius to 0")
      halss_global.radii_ned[idx] = new_radii_ned = 0.
      return halss_global

    halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (320, 320))
    halss_local.surf_norm_thresh(params)
    
    new_radii_local = halss_local.update_landing_site_single(radii_local)
    halss_local.safety_map_circles = plotCircles((160, 160), int(new_radii_local), halss_local.safety_map, 3) # Plot sites on safety map
    if flags.flag_save_pointclouds:
      np.save("Update Landing Site Pointcloud " + str(idx) + ".npy", halss_local.pcd_culled)
    if flags.flag_show_single_safety:
      cv2.imshow("Update Local Safety Map for Landing Site " + str(idx), halss_local.safety_map_circles)
      cv2.imshow("Update Local Surface Normal " + str(idx), halss_local.surf_norm)
    if flags.flag_save_images:
      cv2.imwrite(params.media_save_path + "Update Landing Site Safety Map with Circle " + str(idx) + ".png", halss_local.safety_map_circles)
      cv2.imwrite(params.media_save_path + "Update Landing Site Safety Map " + str(idx) + ".png", halss_local.safety_map)
      cv2.imwrite(params.media_save_path + "Update Landing Site Surface Normal " + str(idx) + ".png", halss_local.surf_norm)
      cv2.imwrite(params.media_save_path + "Update Landing Site Angle Map " + str(idx) + ".png", scale_theta(halss_local.angle_map))
    radii_sf = new_radii_local/radii_local
  new_radii_ned = radius_ned*radii_sf
  t1_update_landing_site_func = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to update landing site: ", t1_update_landing_site_func-t0_update_landing_site_func, "]")
  halss_global.radii_ned[idx] = new_radii_ned
  return halss_global

def multi_update_landing_site_radii(thread_idx, halss_global, params, flags, buffer = 2):
  t0_update_landing_site_func = time.time()
  # new_radii_ned = np.zeros_like(halss_global.radii_ned)
  radii_local = int(160*(1/buffer))
  radius_ned = halss_global.radii_ned[thread_idx]

  #for idx, radius_ned in enumerate(halss_global.radii_ned):
  halss_local = halss_data_packet()
  halss_local.x_cell_size = params.x_cell_size_fine
  halss_local.y_cell_size = params.y_cell_size_fine
  
  dist_squared = (halss_global.pcd_full[:,0] - halss_global.center_coords_ned[thread_idx][0])**2 + (halss_global.pcd_full[:,1] - (-halss_global.center_coords_ned[thread_idx][1]))**2
  within = dist_squared < (buffer*radius_ned)**2
  halss_local.pcd_full = halss_global.pcd_full[within]
  halss_local.pcd_new = halss_local.pcd_full

  halss_local.downsample_pointcloud_dict()
  halss_local.downsample_pointcloud()
  
  if len(halss_local.pcd_culled) > 10:
    if flags.flag_debug:
      print("--> [[RADII UPDATE] DEBUG: Number of points in local pointcloud for Site " + str(thread_idx) + ": " + str(len(halss_local.pcd_culled)) + " out of " + str(maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size)) + " possible points]")
    if len(halss_local.pcd_culled) > maximum_possible_points(halss_local.pcd_full, halss_local.x_cell_size, halss_local.y_cell_size):
      print("--> [[RADII UPDATE] Warning!: Chris can't downsample small pointclouds]")
    halss_local.pc_to_surf_normal(params)
  else:
    print("--> [[RADII UPDATE] Warning!: There are probably not enough points in the landing site to generate a surface normal. Setting Radius to 0]")
    halss_global.radii_ned[thread_idx] = new_radii_ned = 0.
    return 

  if halss_local.surf_norm.max == 0:
    print("--> [[RADII UPDATE] Warning!: Surface normal is all zeros. Setting Radius to 0")
    halss_global.radii_ned[thread_idx] = new_radii_ned = 0.
    return 

  halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (320, 320))
  halss_local.surf_norm_thresh(params)
  
  new_radii_local = halss_local.update_landing_site_single(radii_local)
  halss_local.safety_map_circles = plotCircles((160, 160), int(new_radii_local), halss_local.safety_map, 3) # Plot sites on safety map
  if flags.flag_save_pointclouds:
    np.save("Update Landing Site Pointcloud " + str(thread_idx) + ".npy", halss_local.pcd_culled)
  if flags.flag_show_single_safety:
    cv2.imshow("Update Local Safety Map for Landing Site " + str(thread_idx), halss_local.safety_map_circles)
    cv2.imshow("Update Local Surface Normal " + str(thread_idx), halss_local.surf_norm)
  if flags.flag_save_images:
    cv2.imwrite(params.media_save_path + "Update Landing Site Safety Map with Circle " + str(thread_idx) + ".png", halss_local.safety_map_circles)
    cv2.imwrite(params.media_save_path + "Update Landing Site Safety Map " + str(thread_idx) + ".png", halss_local.safety_map)
    cv2.imwrite(params.media_save_path + "Update Landing Site Surface Normal " + str(thread_idx) + ".png", halss_local.surf_norm)
    cv2.imwrite(params.media_save_path + "Update Landing Site Angle Map " + str(thread_idx) + ".png", scale_theta(halss_local.angle_map))
  radii_sf = new_radii_local/radii_local
  new_radii_ned = radius_ned*radii_sf
  t1_update_landing_site_func = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to update landing site: ", t1_update_landing_site_func-t0_update_landing_site_func, "]")
  halss_global.radii_ned[thread_idx] = new_radii_ned
  return 

def hazard_detection_coarse(halss_global, flags, params):
  t0_full_interp = time.time()
  halss_global.pc_to_surf_normal(params) # Mask out interpolated reigons non-convex reigons of interpolated surface normals
  t1_full_interp = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to interpolate surface normals: ", t1_full_interp-t0_full_interp, "]")

  t0_mask_surf_norm = time.time()
  halss_global.mask_surf_norm() # Mask out interpolated reigons non-convex reigons of interpolated surface normals
  t1_mask_surf_norm = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to mask surface normals: ", t1_mask_surf_norm-t0_mask_surf_norm, "]")
  # #####################
  # Run through Segmentation Network 
  # #####################
  t0_network = time.time()
  halss_global.run_network(params)
  t1_network = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to run network: ", t1_network-t0_network, "]")
  return halss_global

def coarse_landing_region_selection(halss_global, flags, params):
  halss_global = hazard_detection_coarse(halss_global, flags, params)
  t0_landing_selection = time.time()
  halss_global.landing_selection(flags)
  t1_landing_selection = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to select new landing sites: ", t1_landing_selection-t0_landing_selection, "]")

  if flags.flag_show_images:
    cv2.imshow("Global Safety Map with Reigons", cv2.resize(halss_global.safety_map_circles, (2*halss_global.safety_map_circles.shape[1], 2*halss_global.safety_map_circles.shape[0])))
  if flags.flag_save_images:
    cv2.imwrite(params.media_save_path + "Global_Safety_Map_with_Regions.png", halss_global.safety_map_circles)
  
  halss_global.radii_ned = halss_global.sf_x*np.array((halss_global.radii_uv)).astype("float") # Convert new radii to numpy array
  return halss_global

def plotCircles_NED(client, number_of_lines, ned_radius, x_ned_center, y_ned_center, z_ned_center, color=[0, 1, 0, 1]):
  angles = np.linspace(0,2*np.pi,number_of_lines)
  pos_radial_points_ned = []
  for m in range(number_of_lines):
    xr = x_ned_center + ned_radius * np.cos(angles[m])
    yr = y_ned_center + ned_radius * np.sin(angles[m])
    zr = z_ned_center
    if m == 0:
      start = (xr, yr, zr)
      pos_radial_points_ned.append(start)
    elif m > 0 and m < number_of_lines-1:
      new_point = (xr, yr, zr)

      pos_radial_points_ned.append(new_point)
      pos_radial_points_ned.append(new_point)
    elif m == number_of_lines-1:
      new_point = (xr, yr, zr)

      pos_radial_points_ned.append(new_point)
      pos_radial_points_ned.append(new_point)
      pos_radial_points_ned.append(start)
  pos_radial_points_ned = nparray2vector3rlist(np.array(pos_radial_points_ned))
  client.simPlotLineList(pos_radial_points_ned, color, 50, is_persistent=True)

def multithread_update_radii(halss_global, flags, params):
  # Create a thread for each landing site
  threads = []
  for idx in range(halss_global.radii_ned.shape[0]):
    t = threading.Thread(target=multi_update_landing_site_radii, args=(idx, halss_global, params, flags))
    threads.append(t)
    t.start()
    # halss_global = multi_update_landing_site_radii(idx, halss_global, params, flags)
    if flags.flag_debug:
      print("[HALSS: Starting thread to update landing site: ", idx, "]")
  for idx, t in enumerate(threads):
    t.join()
    if flags.flag_debug:
      print("[HALSS: Finished thread to update landing site: ", idx, "]")
  return halss_global

def multithread_fine_site_selection(halss_global, flags, params):
  # Create a thread for each landing site
  threads = []
  for idx in range(halss_global.radii_ned.shape[0]):
    t = threading.Thread(target=multi_fine_landing_site_selection, args=(idx, halss_global, params, flags))
    threads.append(t)
    t.start()
    #halss_global = multi_fine_landing_site_selection(idx, halss_global, params, flags)
    if flags.flag_debug:
      print("[HALSS: Starting thread to update landing site: ", idx, "]")
  for idx, t in enumerate(threads):
    t.join()
    if flags.flag_debug:
      print("[HALSS: Finished thread to update landing site: ", idx, "]")
  halss_global.variance_map_vis = cv2.applyColorMap(halss_global.variance_map_vis, cv2.COLORMAP_INFERNO)
  return halss_global

class halss_data_packet:
  def __init__(self):
    self.pcd_full = np.array([])
    self.pcd_new = np.array([])
    self.pcd_culled_dict = {}
    self.pcd_culled = np.array([])
    self.pcd_x_min = None
    self.pcd_x_max = None
    self.pcd_y_min = None
    self.pcd_y_max = None
    self.org_x = None
    self.org_y = None
    self.sf_x = None
    self.sf_y = None
    self.x_cell_size = None
    self.y_cell_size = None
    self.sn_x_min = None
    self.sn_x_max = None
    self.sn_y_min = None
    self.sn_y_max = None
    self.num_circles = None
    self.center_coords_ned = np.array([])
    self.center_coords_uv = np.array([])
    self.radii_ned = np.array([])
    self.radii_uv = np.array([])
    self.safety_map = np.array([])
    self.variance_map = np.array([])
    self.variance_map_vis = np.array([])
    self.angle_map = np.array([])
    self.surf_norm = np.array([])
    self.safety_map_circles = np.array([])
    self.skeleton = np.array([])
    self.distance_map = np.array([])

  def scale_uv_2_world(self):
    self.pcd_x_min = self.pcd_full[:,0].min()
    self.pcd_x_max = self.pcd_full[:,0].max()
    self.pcd_y_min = self.pcd_full[:,1].min()
    self.pcd_y_max = self.pcd_full[:,1].max()
    self.sf_x = (self.pcd_x_max-self.pcd_x_min)/(self.sn_x_max - self.sn_x_min)
    self.sf_y = (self.pcd_y_max-self.pcd_y_min)/(self.sn_y_max - self.sn_y_min)
  
  def ned2uv(self, x, y):
    self.scale_uv_2_world()
    u = (y / self.sf_y) + self.org_y
    v = (x / self.sf_x) + self.org_x
    return u, v

  def uv2ned(self, u, v, flags):
    del_x = v - self.org_x
    del_y = u - self.org_y
    x_ned_scaled = del_x*self.sf_x
    y_ned_scaled = del_y*self.sf_y

    pcd_kd_tree = scipy.spatial.KDTree(self.pcd_full[:,:2])
    dist, idx = pcd_kd_tree.query(np.array([x_ned_scaled, -y_ned_scaled]))
    if dist > 4:
      print("Warning! Your landing site is probably in a interpolated region where no point could points are present")
    x_ned_actual, y_ned_actual, z_ned_actual = self.pcd_full[idx,:]

    offset = 0
    if flags.flag_offset == True:
      offset = 1
    if self.center_coords_ned.size == 0:
      self.center_coords_ned = np.array([x_ned_actual, -y_ned_actual, -z_ned_actual-offset]).reshape(1,3)
    else:
      self.center_coords_ned = np.append(self.center_coords_ned, np.array([x_ned_actual, -y_ned_actual, -z_ned_actual-offset]).reshape(1,3), axis=0)

  def center_coord_ned_to_uv(self):
    u_vec = np.zeros(len(self.center_coords_ned))
    v_vec = np.zeros(len(self.center_coords_ned))
    for idx, _ in enumerate(self.center_coords_ned): # Convert sites from NED space to UV pixel space for plotting
        u, v = self.ned2uv(self.center_coords_ned[idx][0], self.center_coords_ned[idx][1])
        u_vec[idx] = u
        v_vec[idx] = v
    self.center_coords_uv = np.array([u_vec, v_vec]).T #u_vec.reshape(-1,1), v_vec.reshape(-1,1)

  def downsample_pointcloud_dict(self):
    pcd_culled_dict = self.pcd_culled_dict
    pcd = self.pcd_new
    x_cell_size = self.x_cell_size
    y_cell_size = self.y_cell_size

    for pt in pcd:
        x = int(0.5+pt[0]/x_cell_size)
        y = int(0.5+pt[1]/y_cell_size)
        if not x in pcd_culled_dict:
            pcd_culled_dict[x] = {}
        row = pcd_culled_dict[x]
        if not y in row:
            row[y] = [pt, pt]
        else:
            if pt[2] < row[y][0][2]:
                row[y][0] = pt
            elif pt[2] > row[y][1][2]:
                row[y][1] = pt
    self.pcd_culled_dict = pcd_culled_dict

  def downsample_pointcloud(self):
    # Downsample Point Cloud
    pcd_culled_dict = self.pcd_culled_dict

    count = 0
    for row in pcd_culled_dict:
        count += 2*len(pcd_culled_dict[row])

    pcd_culled = np.zeros((count,3))
    cur = 0
    for row in pcd_culled_dict:
        for cell in pcd_culled_dict[row]:
            pair = pcd_culled_dict[row][cell]
            pcd_culled[cur] = pair[0]
            pcd_culled[cur+1] = pair[1]
            cur += 2
    self.pcd_culled = pcd_culled

  def find_NED_orgin_uv(self):
    # #####################
    # Find NED Origin in UV Pixel Space
    # #####################
    self.scale_uv_2_world()
    x_pcd_to_surface = self.sn_x_min - (self.sn_x_max-self.sn_x_min)/(self.pcd_x_max- self.pcd_x_min) * (self.pcd_x_min)
    y_pcd_to_surface = self.sn_y_min + (self.sn_y_max-self.sn_y_min) * (0 - self.pcd_y_min)/(self.pcd_y_max - self.pcd_y_min)

    self.org_x = x_pcd_to_surface
    self.org_y = self.sn_y_max - y_pcd_to_surface
    
  def pc_to_surf_normal(self, params):
    model_x_data = np.linspace(min(self.pcd_culled[:,0]), max(self.pcd_culled[:,0]), params.grid_res)
    model_y_data = np.linspace(min(self.pcd_culled[:,1]), max(self.pcd_culled[:,1]), params.grid_res)
    X, Y = np.meshgrid(model_x_data, model_y_data)

    
    f_linear = interpolate.LinearNDInterpolator(list(zip(self.pcd_culled[:,0], self.pcd_culled[:,1])), self.pcd_culled[:,2])
    Z = f_linear(X, Y)
    
    surf_norm_img = surface_normal_from_interp_model(model_x_data, model_y_data, Z, params)

    self.surf_norm = cv2.cvtColor(surf_norm_img, cv2.COLOR_BGR2RGB)

  def run_network(self, params):
    current = normalize(self.surf_norm.transpose(2,0,1))
    kernel_size = 3
    nbase = [3, 32, 64, 128, 256]  # Number of channels per layer
    nout = 1  # Number of outputs

    net = Unet_drop(nbase, nout, kernel_size)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('--> [HALSS: Using ', torch.cuda.get_device_name(0), ' for learning based Coarse HD]')

    print(os.getcwd())
    net.load_model(f"HALSS_utils\\network_utils\\unet_epoch6.pth")
    net.train()
    net.to(device);  # remove semi-colon to see net structure
    
    img_torch = torch.from_numpy(current).to(device).unsqueeze(0)  # also need to add a first dimension

    # Run the network using MC Dropout
    t0_new = time.time()
    img_torch = torch.repeat_interleave(img_torch, params.num_mc_samples, dim = 0)
    out = net(img_torch)
    tf_new = time.time() - t0_new
    # print("Full predicition run time: ", tf_new)
    mean_out = torch.mean(out,0)
    var_out = torch_to_numpy(torch.var(out,0))

    # Build safety map
    predicted_map = 255*(mean_out[0].detach().cpu()).numpy()
    color_safety_map = np.zeros((3,params.grid_res,params.grid_res))
    color_safety_map[0] = predicted_map
    color_safety_map[1] = predicted_map
    color_safety_map[2] = predicted_map
    color_safety_map = color_safety_map.transpose(1,2,0)
    safety_map = scale_image(color_safety_map, self.pcd_full)
    safety_map[safety_map != 255] = 0

    # Build Uncertainty Map
    norm_var_out = normalize(var_out)
    flipped_var = np.flipud(var_out)
    var_scaled = scale_image(flipped_var, self.pcd_full)
    norm_flipped_var = np.flipud(norm_var_out)
    norm_var_scaled = scale_image(norm_flipped_var, self.pcd_full)
    norm_resized_var = cv2.resize(norm_flipped_var, (1*norm_var_scaled.shape[1],1*norm_var_scaled.shape[0]))

    self.safety_map = safety_map
    self.variance_map = var_scaled
    self.variance_map_vis = (np.flipud(norm_resized_var)*255).astype(np.uint8)
    
    # Threshold Safety Map
    thresh_var = self.variance_map > params.thresh
    thresh_safety = cv2.bitwise_and(self.safety_map, self.safety_map, mask = np.invert(thresh_var).astype(np.uint8))
    thresh_safety[thresh_safety != 255] = 0 # Binarize Safety Map
    self.safety_map = thresh_safety
  
  def surf_norm_thresh(self, params):
    flat_vec = np.array([1,0,0])
    norm_png = self.surf_norm.astype(np.float16)/255

    cos_theta_vec = flat_vec @ norm_png.reshape(norm_png.shape[0]*norm_png.shape[1],3).T
    cos_theta_mat = cos_theta_vec.reshape(norm_png.shape[0], norm_png.shape[1])
    theta_mat = np.arccos(cos_theta_mat)*180/np.pi
    safety_map = np.zeros_like(self.surf_norm[:,:,0])
    safety_map[theta_mat < params.alpha] = 255
    safety_map_color = np.zeros((320,320,3))
    safety_map_color[:,:,0] = safety_map
    safety_map_color[:,:,1] = safety_map
    safety_map_color[:,:,2] = safety_map
    self.safety_map = safety_map_color.astype(np.uint8)
    self.angle_map = theta_mat

  def mask_surf_norm(self, scale_factor = 2):
    surf_norm = self.surf_norm
    x_cell_size = self.x_cell_size
    y_cell_size = self.y_cell_size

    masked_surf_norm = np.zeros_like(surf_norm)

    self.sn_x_min = 0
    self.sn_x_max = self.surf_norm.shape[1]
    self.sn_y_min = 0
    self.sn_y_max = self.surf_norm.shape[0]

    x_keys = []
    y_keys = []
    for x_key in list(self.pcd_culled_dict.keys()):
      for y_key in list(self.pcd_culled_dict[x_key].keys()):
        x_keys.append(x_cell_size*x_key)
        y_keys.append(y_cell_size*y_key)

    self.find_NED_orgin_uv()

    u_vec, v_vec = self.ned2uv(x_keys, y_keys)
    box_x_min, box_y_min = self.ned2uv(0, 0)
    box_x_max, box_y_max = self.ned2uv(x_cell_size/2, y_cell_size/2)

    cell_width = (box_x_max - box_x_min)*scale_factor
    cell_height = (box_y_max - box_y_min)*scale_factor
    
    for idx in range(len(u_vec)):
      u = u_vec[idx] - (2*self.org_y - self.sn_x_max)
      v = v_vec[idx]
      v1 = int(0.5+(v - cell_height))
      u1 = int(0.5+(u - cell_width))
      v2 = int(0.5+(v + cell_height))
      u2 = int(0.5+(u + cell_width))
      masked_surf_norm = cv2.rectangle(masked_surf_norm, (v1, u1), (v2, u2), (255, 255, 255), -1)
    masked_surf_norm = np.flipud(masked_surf_norm)
    self.surf_norm = cv2.bitwise_and(surf_norm, surf_norm, mask = masked_surf_norm[:,:,0])

  def landing_selection(self, flags):
    data = prep_safety_mask(self.safety_map)
    
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)
    dist_on_skel = distance * skel


    center_coords, radii = topNCircles(distance, self.num_circles)
    u_vec = np.zeros(len(center_coords))
    v_vec = np.zeros(len(center_coords))
    for i in range(len(center_coords)):
        u_vec[i] = center_coords[i][0][0]
        v_vec[i] = center_coords[i][1][0]

    circles = plotCircles((u_vec, v_vec), radii, data)
    self.radii_uv = radii
    self.skeleton = dist_on_skel
    self.safety_map_circles = circles
    self.distance = distance
    for idx, _ in enumerate(center_coords):
      self.uv2ned(center_coords[idx][0][0], center_coords[idx][1][0], flags)
  
  def percep2traj(self, scores, alt):
    landing_site_save = np.zeros((len(self.radii_ned),4)) # Initialize pack to send to traj planner to store new sites
    landing_site_save[:,0:3] = self.center_coords_ned  # Store new sites in data packet
    landing_site_save[:,3] = self.radii_ned # Store new radii in data packet
    output = np.hstack((landing_site_save, scores.T))
    print("--> [HALSS: I am sending back ", output.shape[0], " landing sites]")
    np.save(os. getcwd() + '\\AirSim\\temp\\percep_to_traj.npy', output) # Save data packet to file
    np.save(os. getcwd() + '\\AirSim\\temp\\percep_to_traj_alt.npy', np.array([alt])) # Save data packet to file
    print("--> [PUB @ ", datetime.now().strftime("%H:%M:%S.%f"), "]: ")
  
  def traj2percep(self, path_input):
    input = np.load(path_input)
    if len(input) == 0:
      self.radii_ned = np.array([])
      self.center_coords_ned = np.array([])
      print("--> [HALSS: I Recieved ", len(input), " targets]")
      return
    if len(input.shape) == 1:
        input = np.expand_dims(input, axis=0)
    self.radii_ned = input[:,3] # Extract locked radii from data packet
    self.center_coords_ned = input[:,0:3] # Extract locked sites from data packet
    print("--> [HALSS: I Recieved ", len(input), " targets]")
    print("--> [SUB @ ", datetime.now().strftime("%H:%M:%S.%f"), "]: ")


  def plot_circles_unreal(self, client):
    landing_site_positions = []
    number_of_lines = 50 # Number of lines to plot for each circle
    for k in range(len(self.radii_ned)):
      landing_site_positions.append(Vector3r(-self.center_coords_ned[k][1], self.center_coords_ned[k][0], self.center_coords_ned[k][2]))
      plotCircles_NED(client, number_of_lines, self.radii_ned[k], -self.center_coords_ned[k][1], self.center_coords_ned[k][0], self.center_coords_ned[k][2])
    client.simPlotPoints(landing_site_positions, color_rgba=[0,1,0,1], size = 5.0, is_persistent = True)
  
  def update_landing_site_single(self, radius):
    # If there is an unsafe region within the landing site, then resize the radius of the landing site
    safety_map_flipped = binarize(self.safety_map)

    new_radius = 0
      
    int_radius = int(radius)
    safe_flip_cp = np.copy(safety_map_flipped)
    mask = np.zeros((safe_flip_cp.shape[0], safe_flip_cp.shape[1]), dtype = "uint8")
    cv2.circle(mask, (int(self.safety_map.shape[0]/2), int(self.safety_map.shape[0]/2)), int_radius, 255, -1)
    masked_safety_map = cv2.bitwise_and(safe_flip_cp, safe_flip_cp, mask=mask)
    max_x = int(self.safety_map.shape[0]/2 + radius)
    min_x = int(self.safety_map.shape[0]/2 - radius)
    max_y = int(self.safety_map.shape[0]/2 + radius)
    min_y = int(self.safety_map.shape[0]/2 - radius)

    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2

    off_min_x = 0 if min_x < 0 else min_x
    off_min_y = 0 if min_y < 0 else min_y
    off_max_x = masked_safety_map.shape[0] if max_x > masked_safety_map.shape[0] else max_x
    off_max_y = masked_safety_map.shape[1] if max_y > masked_safety_map.shape[1] else max_y

    rel_center_x = center_x - off_min_x
    rel_center_y = center_y - off_min_y

    landing_site = masked_safety_map[off_min_x:off_max_x, off_min_y:off_max_y]
    landing_site = np.invert(landing_site.astype(np.uint8))
    unsafe_pixels = np.nonzero(landing_site)
    try:
      new_radius = np.sqrt((rel_center_x - unsafe_pixels[0])**2 + (rel_center_y - unsafe_pixels[1])**2).min()
    except ValueError:
      new_radius = radius

    return new_radius

      
class flags_required:
  def __init__(self):
    self.flag_plot_origin = None
    self.flag_plot_circles = None
    self.flag_show_images = None
    self.flag_save_images = None
    self.flag_timing = None
    self.flag_show_single_safety = None
    self.flag_show_single_safety_fine = None
    self.flag_debug = None
    self.flag_save_pointclouds = None
    self.flag_offset = None

class parameters:
  def __init__(self):
    self.grid_res = None
    self.x_cell_size_coarse = None
    self.y_cell_size_coarse = None
    self.x_cell_size_fine = None
    self.y_cell_size_fine = None
    self.alpha = None
    self.max_sites = None
    self.thresh = None
    self.num_mc_samples = None
    self.num_circles = None
    self.media_save_path = None