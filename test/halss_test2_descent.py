#################
# Imports
#################
from pdb import set_trace as debug

# Import the repository root directory (one level above the ros root directory)
import sys,os
root_path = os.path.abspath(os.getcwd() + "/../..")
sys.path.append(root_path)

# Other
import sys
import numpy as np
import time
import os
import cv2
from matplotlib import cm


# Set randomization seed
np.random.seed(0)

# Test parameters
radii_start = 100
radii_declimation = .25
p_noisy = 0.00001 # probability of noisy reading
reacquire_radii = [75,50,25]

#################
# Create sample PCD buffer
#################
# Create a circular pointcloud confined to a 2D plane in 3D space (X-Y plane) with a radius of 100m
rad_pointcloud = radii_start
num_channels = 128
num_angles = 512
angles = np.linspace(0, 2*np.pi, num_angles)
channels = np.linspace(0, rad_pointcloud, num_channels)
num_points = num_channels*num_angles
pcd_buffer = np.zeros((num_points, 3))
for i, angle in enumerate(angles):
    for j, channel in enumerate(channels):
        pcd_buffer[i*num_channels+j, 0] = channel*np.cos(angle)
        pcd_buffer[i*num_channels+j, 1] = channel*np.sin(angle)
        pcd_buffer[i*num_channels+j, 2] = 0
        
# Create random perturbations in the Z direction to simulate noisy readings
p_noisy_init = 0.001
z_perturb = 1 # meter
z_noise = np.random.uniform(0,1,pcd_buffer.shape[0])
z_noise = np.where(z_noise < p_noisy_init, z_perturb, 0)
pcd_buffer[:,2] += z_noise

#################
# Setup: HALSS
#################
from HALSS.classes import *
from HALSS.algorithms import *
from matplotlib import cm

# Set HALSS flags (not in use currently)
flags = flags_required()

# Set HALSS parameters
params = parameters()
params.x_cell_size_coarse = 1 # Size of cell in x direction for Coarse Downsampling
params.y_cell_size_coarse = 1 # Size of cell in y direction for Coarse Downsampling
params.x_cell_size_fine   = 1 # Size of cell in x direction for Fine Downsampling
params.y_cell_size_fine   = 1 # Size of cell in y direction for Fine Downsampling
params.grid_res = 320 # Resolution of grid to be used for segmentation
params.alpha = 8 # Maximum allowable inclination angle in degrees
params.max_sites = 4 # Maximum number of sites to be considered

# Build (unpopulated) HALSS data packet
packet = halss_data_packet()
packet.type = 'global'
packet.num_sites = params.max_sites
packet.x_cell_size = params.x_cell_size_coarse # Size of cell in x direction for Coarse Downsampling
packet.y_cell_size = params.y_cell_size_coarse # Size of cell in y direction for Coarse Downsampling
packet.pcd_raw = pcd_buffer

# Build additional HALSS data packets for each coarse landing region.
packets_fine = [halss_data_packet() for _ in range(packet.num_sites)]

# Coloring
colors_targets = cm.get_cmap('gist_rainbow')(np.linspace(0,1,params.max_sites)).tolist()
colors_targets = [(int(255*c[2]),int(255*c[1]),int(255*c[0])) for c in colors_targets] # convert to 0-255 BGR format for CV2
image_resize_factor = 2

#################
# HALSS: Initial Target Acquisition
#################
# Perform pointcloud downsampling
packet.downsample_pointcloud()

# Perform coarse landing region selection
packet = coarse_landing_region_selection(packet, flags, params)

# Perform fine landing site selection
for site_idx in range(packet.num_sites):
    packet, packets_fine[site_idx] = fine_landing_site_selection(site_idx, packet, flags, params)

#################
# Simulation: Decimate radii (simulate descent)
#################
radii = radii_start
center_zoom = np.array([0,0])
test_iter = 0
while radii > 0:
    #################
    # Modify PCD buffer by zooming in on 2D region of smaller radius
    #################
    new_rad_pointcloud = radii # meters
    dist_squared = (pcd_buffer[:,0] - center_zoom[0])**2 + (pcd_buffer[:,1] - center_zoom[1])**2
    within = dist_squared < new_rad_pointcloud**2
    pcd_buffer = pcd_buffer[within]
    
    # Create random perturbations in the Z direction to simulate noisy readings
    z_perturb = 1 # meter
    z_noise = np.random.uniform(0,1,pcd_buffer.shape[0])
    z_noise = np.where(z_noise < p_noisy, z_perturb, 0)
    pcd_buffer[:,2] += z_noise

    #################
    # HALSS: Scheduled Target Re-acquisition
    #################
    if len(reacquire_radii) > 0:
        if radii < reacquire_radii[0]:
            # Perform pointcloud downsampling
            packet.downsample_pointcloud()

            # Perform coarse landing region selection
            packet = coarse_landing_region_selection(packet, flags, params)

            # Perform fine landing site selection
            for site_idx in range(packet.num_sites):
                packet, packets_fine[site_idx] = fine_landing_site_selection(site_idx, packet, flags, params)
                
            reacquire_radii.pop(0)

    #################
    # HALSS: Target Update
    #################
     # Set packet PCD to the buffered PCD
    packet.pcd_raw = pcd_buffer.copy()

    # Perform pointcloud downsampling
    packet.downsample_pointcloud()

    # Update the radii of each target
    for site_idx in range(packet.num_sites):
        packet, packets_fine[site_idx] = update_landing_site(site_idx, packet, packets_fine[site_idx], flags, params)

    # Update global safety map for plotting purposes only
    packet = update_global_safety_map(packet, flags, params)

    #################
    # HALSS: Plotting
    #################
    resize_fun = lambda dims,fac : (int(dims[1]*fac), int(dims[0]*fac))

    # Global Safety Map
    prep_safety_map = prep_safety_mask(packet.safety_map) # Prepare safety map for plotting 
    safety_map_global = plotCircles((packet.center_coords_uv_coarse[:,0], packet.center_coords_uv_coarse[:,1]), packet.radii_uv_coarse, prep_safety_map, colors=colors_targets, fill_frac=10, border=False, center=False) # Plot coarse regions on safety map
    safety_map_global = plotCircles((packet.center_coords_uv[:,0], packet.center_coords_uv[:,1]), packet.radii_uv, safety_map_global, colors=colors_targets) # Plot sites on safety map
    safety_map_global = cv2.resize(safety_map_global, resize_fun(safety_map_global.shape, image_resize_factor))

    # Local safety map(s)
    safety_maps_local = []
    for (idx,packet_) in enumerate(packets_fine):
        prep_safety_map = prep_safety_mask(packet_.safety_map) # Prepare safety map for plotting
        prep_safety_map = plotCircles((params.grid_res//2, params.grid_res//2), params.grid_res//2, prep_safety_map, colors=colors_targets[idx], fill_frac=10, border=False, center=False) # Plot coarse regions on safety map
        prep_safety_map = plotCircles((packet_.center_coords_uv[:,0], packet_.center_coords_uv[:,1]), packet_.radii_uv, prep_safety_map, colors=colors_targets[idx]) # Plot sites on safety map
        prep_safety_map = cv2.resize(prep_safety_map, resize_fun(prep_safety_map.shape, image_resize_factor//2))
        safety_maps_local.append(prep_safety_map)

    # Construct tiled map
    int_map1 = cv2.hconcat([safety_maps_local[0], safety_maps_local[1]])
    int_map2 = cv2.hconcat([safety_maps_local[2], safety_maps_local[3]])
    int_map3 = cv2.vconcat([int_map1, int_map2])
    composite_map = cv2.hconcat([safety_map_global, int_map3])

    # Display
    cv2.imshow('Composite Safety Maps', composite_map)
    cv2.waitKey(10)
    
    # Configure video
    if test_iter == 0:
        # Video setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('halss_test2_descent.mp4', fourcc, 30, composite_map.shape[:2][::-1])
    video.write(composite_map)
    
    # Decimate radii
    radii -= radii_declimation
    test_iter += 1

video.release()
print("Video saved as 'halss_test2_descent.avi'")