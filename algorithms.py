# Import the repository root directory (one level above the ros root directory)
import sys,os
root_path = os.path.abspath(os.getcwd() + "/..")
sys.path.append(root_path)

# Base imports
import cosysairsim as airsim
import numpy as np
import cv2
from scipy import interpolate
import scipy.ndimage
import time

# Custom imports
from HALSS.utils.utils import maximum_possible_points
from HALSS.classes import *

def hazard_detection_coarse(halss_global, flags, params):
  # Builds up the coarse-HD map with hazards detected
  
  # Obtain surface normal map from pointcloud
  t0_full_interp = time.time()
  halss_global.pc_to_surf_normal(params)
  t1_full_interp = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to interpolate surface normals: ", t1_full_interp-t0_full_interp, "]")

  # Mask out non-convex regions of interpolated surface normals
  t0_mask_surf_norm = time.time()
  halss_global.mask_surf_norm()
  t1_mask_surf_norm = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to mask surface normals: ", t1_mask_surf_norm-t0_mask_surf_norm, "]")
  
  # Threshold the surface normal map
  t0_threshold = time.time()
  halss_global.surf_norm_thresh(params)
  halss_global.surf_norm = scale_image(halss_global.surf_norm, halss_global.pcd_global)
  t1_threshold = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to perform thresholding: ", t1_threshold-t0_threshold, "]")
    
  return halss_global

def coarse_landing_region_selection(halss_global, flags, params):
  # Performs landing site selection for the coarse hazard detection algorithm
  
  # Build the map
  halss_global = hazard_detection_coarse(halss_global, flags, params)
  
  # Select landing sites (regions)
  t0_landing_selection = time.time()
  halss_global.landing_selection(flags)
  halss_global.radii_ned = halss_global.sf_x*np.array((halss_global.radii_uv)).astype("float") # Convert new radii to numpy array
  t1_landing_selection = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to select new landing sites: ", t1_landing_selection-t0_landing_selection, "]")
  
  return halss_global

def fine_landing_site_selection(site_idx, halss_global, flags, params, buffer = 1.1):
  new_center_coords_ned = np.zeros_like(halss_global.center_coords_ned[0])

  halss_local = halss_data_packet()
  halss_local.num_circles = 1
  halss_local.x_cell_size = params.x_cell_size_fine
  halss_local.y_cell_size = params.y_cell_size_fine
  radius_ned = halss_global.radii_ned[site_idx]

  dist_squared = (halss_global.pcd_global[:,0] - halss_global.center_coords_ned[site_idx][0])**2 + (halss_global.pcd_global[:,1] - (-halss_global.center_coords_ned[site_idx][1]))**2
  within = dist_squared < (buffer*radius_ned)**2
  halss_local.pcd_global = halss_global.pcd_global[within]
  halss_local.pcd_local = halss_local.pcd_global

  halss_local.downsample_pointcloud_dict()
  halss_local.downsample_pointcloud()
  
  if len(halss_local.pcd_culled) > 10:
    if flags.flag_debug:
      print("--> [[FINE SELECTION] DEBUG: Number of points in local pointcloud for Site " + str(site_idx) + ": " + str(len(halss_local.pcd_culled)) + " out of " + str(maximum_possible_points(halss_local.pcd_global, halss_local.x_cell_size, halss_local.y_cell_size)) + " possible points]")
    if len(halss_local.pcd_culled) > maximum_possible_points(halss_local.pcd_global, halss_local.x_cell_size, halss_local.y_cell_size):
      print("--> [[FINE SELECTION] Warning!: Chris lost his mind downsampling small pointclouds]")
    halss_local.pc_to_surf_normal(params)
  else:
    print("--> [[FINE SELECTION] Warning!: There are probably not enough points in the landing site to generate a surface normal. Setting Radius to 0]")
    halss_local.radii_ned = 0.
    return halss_global

  halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (params.grid_res, params.grid_res))

  halss_local.surf_norm_thresh(params)

  halss_local.sn_x_min = 0
  halss_local.sn_x_max = halss_local.safety_map.shape[1]
  halss_local.sn_y_min = 0
  halss_local.sn_y_max = halss_local.safety_map.shape[0]
  halss_local.find_NED_origin_uv()

  halss_local.landing_selection(flags)
  
  if flags.flag_save_pointclouds:
    np.save(params.media_save_path + "Fine Landing Site Pointcloud " + str(site_idx) + ".npy", halss_local.pcd_culled)
  if flags.flag_show_single_safety_fine:
    cv2.imshow("Fine Landing Site Safety Map with Circle" + str(site_idx), halss_local.safety_map_circles)
    cv2.imshow("Fine Landing Site Surface Normal " + str(site_idx), halss_local.surf_norm)
    cv2.imshow("Fine Angle Map " + str(site_idx), scale_theta(halss_local.angle_map))
  if flags.flag_save_images:
    cv2.imwrite(params.media_save_path + "Fine Landing Site Safety Map with Circle " + str(site_idx) + ".png", halss_local.safety_map_circles)
    cv2.imwrite(params.media_save_path + "Fine Landing Site Safety Map " + str(site_idx) + ".png", halss_local.safety_map)
    cv2.imwrite(params.media_save_path + "Fine Landing Site Surface Normal " + str(site_idx) + ".png", halss_local.surf_norm)
    cv2.imwrite(params.media_save_path + "Fine Angle Map " + str(site_idx) + ".png", scale_theta(halss_local.angle_map))
    cv2.imwrite(params.media_save_path + "Fine Skeleton " + str(site_idx) + ".png", halss_local.skeleton)
  halss_local.radii_ned = halss_local.sf_x*np.array((halss_local.radii_uv)).astype("float") # Convert new radii to numpy array
  new_center_coords_ned = np.squeeze(halss_local.center_coords_ned)
  halss_global.radii_ned[site_idx] = halss_local.radii_ned  
  halss_global.center_coords_ned[site_idx] = new_center_coords_ned.T
  halss_global.halss_locals.append(halss_local)
  return halss_global

def update_landing_site_radii(site_idx, halss_global, flags, params, buffer = 2):
  t0_update_landing_site_func = time.time()
  # new_radii_ned = np.zeros_like(halss_global.radii_ned)
  radii_local = int(params.grid_res/2*(1/buffer))
  radius_ned = halss_global.radii_ned[site_idx]

  #for idx, radius_ned in enumerate(halss_global.radii_ned):
  halss_local = halss_data_packet()
  halss_local.x_cell_size = params.x_cell_size_fine
  halss_local.y_cell_size = params.y_cell_size_fine
  
  dist_squared = (halss_global.pcd_global[:,0] - halss_global.center_coords_ned[site_idx][0])**2 + (halss_global.pcd_global[:,1] - (-halss_global.center_coords_ned[site_idx][1]))**2
  within = dist_squared < (buffer*radius_ned)**2
  halss_local.pcd_global = halss_global.pcd_global[within]
  halss_local.pcd_local = halss_local.pcd_global

  halss_local.downsample_pointcloud_dict()
  halss_local.downsample_pointcloud()
  
  if len(halss_local.pcd_culled) > 10:
    if flags.flag_debug:
      print("--> [[RADII UPDATE] DEBUG: Number of points in local pointcloud for Site " + str(site_idx) + ": " + str(len(halss_local.pcd_culled)) + " out of " + str(maximum_possible_points(halss_local.pcd_global, halss_local.x_cell_size, halss_local.y_cell_size)) + " possible points]")
    if len(halss_local.pcd_culled) > maximum_possible_points(halss_local.pcd_global, halss_local.x_cell_size, halss_local.y_cell_size):
      print("--> [[RADII UPDATE] Warning!: Chris can't downsample small pointclouds]")
    halss_local.pc_to_surf_normal(params)
  else:
    print("--> [[RADII UPDATE] Warning!: There are probably not enough points in the landing site to generate a surface normal. Setting Radius to 0]")
    halss_global.radii_ned[site_idx] = new_radii_ned = 0.
    return 

  if halss_local.surf_norm.max == 0:
    print("--> [[RADII UPDATE] Warning!: Surface normal is all zeros. Setting Radius to 0")
    halss_global.radii_ned[site_idx] = new_radii_ned = 0.
    return 

  halss_local.surf_norm = cv2.resize(halss_local.surf_norm, (params.grid_res, params.grid_res))
  halss_local.surf_norm_thresh(params)
  
  new_radii_local = halss_local.update_landing_site_single(radii_local)
  halss_local.safety_map_circles = plotCircles((params.grid_res/2, params.grid_res/2), int(new_radii_local), halss_local.safety_map, 3) # Plot sites on safety map
  if flags.flag_save_pointclouds:
    np.save("Update Landing Site Pointcloud " + str(site_idx) + ".npy", halss_local.pcd_culled)
  if flags.flag_show_single_safety:
    cv2.imshow("Update Local Safety Map for Landing Site " + str(site_idx), halss_local.safety_map_circles)
    cv2.imshow("Update Local Surface Normal " + str(site_idx), halss_local.surf_norm)
  if flags.flag_save_images:
    cv2.imwrite(params.media_save_path + "Update Landing Site Safety Map with Circle " + str(site_idx) + ".png", halss_local.safety_map_circles)
    cv2.imwrite(params.media_save_path + "Update Landing Site Safety Map " + str(site_idx) + ".png", halss_local.safety_map)
    cv2.imwrite(params.media_save_path + "Update Landing Site Surface Normal " + str(site_idx) + ".png", halss_local.surf_norm)
    cv2.imwrite(params.media_save_path + "Update Landing Site Angle Map " + str(site_idx) + ".png", scale_theta(halss_local.angle_map))
  radii_sf = new_radii_local/radii_local
  new_radii_ned = radius_ned*radii_sf
  t1_update_landing_site_func = time.time()
  if flags.flag_timing:
    print("--> [TIMING: Time to update landing site: ", t1_update_landing_site_func-t0_update_landing_site_func, "]")
  halss_global.radii_ned[site_idx] = new_radii_ned
  return

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
    dist = (halss_data.pcd_global[:,0] - halss_data.center_coords_ned[idx][0])**2 + (halss_data.pcd_global[:,1] - (-halss_data.center_coords_ned[idx][1]))**2
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