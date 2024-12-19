# Import the repository root directory (one level above the ros root directory)
import sys,os
root_path = os.path.abspath(os.getcwd() + "/..")
sys.path.append(root_path)

# Base imports
import numpy as np
import cv2
import scipy.ndimage

# Custom imports
from HALSS.classes import *

def coarse_landing_region_selection(halss_global, flags, params):
  # Performs landing site selection for the coarse hazard detection algorithm
  
  # Obtain surface normal map from pointcloud
  halss_global.pc_to_surf_normal(params)

  # Mask out non-convex regions of interpolated surface normals
  halss_global.mask_surf_norm()
  
  # Threshold the surface normal map
  halss_global.surf_norm_to_safety_map(params)
  
  # Select landing regions
  halss_global.find_landing_site()
  
  # Store coarse region information for future reference
  halss_global.center_coords_ned_coarse = halss_global.center_coords_ned.copy()
  halss_global.center_coords_uv_coarse = halss_global.center_coords_uv.copy()
  halss_global.radii_ned_coarse = halss_global.radii_ned.copy()
  halss_global.radii_uv_coarse = halss_global.radii_uv.copy()
  
  return halss_global

def fine_landing_site_selection(site_idx, halss_global, flags, params):
  # Construct a data packet associated with the local landing region
  halss_local = halss_data_packet()
  halss_local.num_sites = 1
  halss_local.x_cell_size = params.x_cell_size_fine
  halss_local.y_cell_size = params.y_cell_size_fine
  
  # Determine the local pointcloud based on the region location
  halss_local = region_localize(site_idx, halss_global, halss_local)

  # Check if there are enough points in the local pointcloud to generate a surface normal
  if site_check(site_idx, halss_local, flags, params):
    halss_local.pc_to_surf_normal(params)
  else:
    return halss_global, halss_local

  # Generate the adjusted surface normal map with thresholding
  halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (params.grid_res, params.grid_res))
  halss_local.surf_norm_to_safety_map(params)
  halss_local.find_NED_origin_uv() # locates the NED origin in the UV frame
  
  # Perform the fine landing site selection
  halss_local.find_landing_site()
  
  # Global packet updates
  halss_global.center_coords_ned[site_idx] = halss_local.center_coords_ned[0] 
  halss_global.center_coords_ned_to_uv(site_idx)
  halss_global.radii_ned[site_idx] = halss_local.radii_ned[0]
  halss_global.radii_uv[site_idx]  = halss_local.radii_ned[0]/halss_global.sf_x
  
  return halss_global, halss_local

def update_landing_site(site_idx, halss_global, halss_local, flags, params):
  # Determine the local pointcloud based on the region location
  halss_local = region_localize(site_idx, halss_global, halss_local)
  
  # Check if there are enough points in the local pointcloud to generate a surface normal
  if site_check(site_idx, halss_local, flags, params):
    halss_local.pc_to_surf_normal(params)
  else:
    return halss_global, halss_local
  
  # Generate the adjusted surface normal map with thresholding
  halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (params.grid_res, params.grid_res))
  halss_local.surf_norm_to_safety_map(params)
  halss_local.find_NED_origin_uv() # locates the NED origin in the UV frame
  
  # Update radius of the landing site
  halss_local.update_landing_site(params)
  
  # Global packet updates (used for plotting on the global safety map)
  halss_global.center_coords_ned[site_idx] = halss_local.center_coords_ned[0] 
  halss_global.center_coords_ned_to_uv(site_idx)
  halss_global.radii_ned[site_idx] = halss_local.radii_ned[0]
  halss_global.radii_uv[site_idx]  = halss_local.radii_ned[0]/halss_global.sf_x
  
  return halss_global, halss_local

def update_global_safety_map(halss_global, flags, params):
  # Obtain surface normal map from pointcloud
  halss_global.pc_to_surf_normal(params)

  # Mask out non-convex regions of interpolated surface normals
  halss_global.mask_surf_norm()
  
  # Threshold the surface normal map
  halss_global.surf_norm_to_safety_map(params)

  # Update the coarse region information in UV frame
  for site_idx in range(halss_global.num_sites):
    halss_global.center_coords_ned_to_uv_coarse(site_idx)
    halss_global.radii_uv_coarse[site_idx] = halss_global.radii_ned_coarse[site_idx]/halss_global.sf_x
  
  return halss_global

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
  radius = np.array(halss_data.radii_ned)
  size_scores = radius / radius.max()

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
    dist = (halss_data.pcd_global[:,0] - halss_data.center_coords_ned[idx][0])**2 + (halss_data.pcd_global[:,1] - halss_data.center_coords_ned[idx][1])**2
    within = dist < radius_ned[idx]**2
    if radius_ned[idx] == 0:
      density_score[idx] = 0
    else:
      density_score[idx] = np.sum(within)/(np.pi*radius_ned[idx]**2)
  if density_score.max() == 0:
    density_score = np.ones_like(radius)
  else:
    density_score = density_score/density_score.max()

  # 4. How large is the 99th percentile of the uncertainty map in the landing site
  uncertainty_score = np.ones_like(radius) # deprecated feature (no uncertainty map supported currently)
    
  # 5. How near a landing site is to other landing sites
  prox_map = scipy.spatial.distance.cdist(halss_data.center_coords_ned, halss_data.center_coords_ned, metric='euclidean')
  prox_score = np.sum(prox_map, axis = 0)
  prox_score = 1/(prox_score/prox_score.max())
  prox_score = prox_score/prox_score.max()

  return np.vstack((size_scores, drone_scores, density_score, uncertainty_score, prox_score))