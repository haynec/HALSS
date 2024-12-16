# Import the repository root directory (one level above the ros root directory)
import sys,os
root_path = os.path.abspath(os.getcwd() + "/..")
sys.path.append(root_path)

# Base imports
import cv2
import numpy as np
from scipy import interpolate
import scipy.ndimage
from skimage.morphology import medial_axis
from pdb import set_trace as debug

# Custom imports
from HALSS.utils.utils import *
from HALSS.utils.plotting import *

# Sest HALSS flags
class flags_required:
  def __init__(self):
    self.flag_debug = None          # Higher verbosity level for debugging
    
# Sets HALSS parameters
class parameters:
  def __init__(self):
    self.x_cell_size_coarse = None  # [m] Size of each cell in the x-direction storing height data for coarse landing region selection
    self.y_cell_size_coarse = None  # [m] Size of each cell in the y-direction storing height data for coarse landing region selection
    self.x_cell_size_fine = None    # [m] Size of each cell in the x-direction storing height data for fine landing site selection
    self.y_cell_size_fine = None    # [m] Size of each cell in the y-direction storing height data for fine landing site selection
    self.grid_res = None            # [px] Resolution of the square grid (res x res) for the interpolated surface normal map
    self.alpha = None               # [deg] Maximum elevation angle (surface normal angle from the vertical) to declare a pixel safe in the safety map
    self.max_sites = None           # Maximum number of landing sites to be selected
    self.media_save_path = None     # Path to save the media files

# Packet holding data associated with HALSS global and local site and map information
# Frames:
#     NED: North-East-Down world frame
#     UV: Image frame
class halss_data_packet:
  def __init__(self):
    self.pcd_global = np.array([])               # [m] Global point cloud data
    self.pcd_binned = np.array([])               # [m] PCD processing intermediate step: bins the x-y values according to self.x_cell_size and self.y_cell_size
    self.pcd_culled = np.array([])               # [m] Final downsampled (culled) and binned PCD
    self.pcd_x_min = None                        # [m] Minimum x-value in culled PCD data
    self.pcd_x_max = None                        # [m] Maximum x-value in culled PCD data
    self.pcd_y_min = None                        # [m] Minimum y-value in culled PCD data
    self.pcd_y_max = None                        # [m] Maximum y-value in culled PCD data
    self.org_x = None                            # [px] Origin of the UV frame in the x-direction
    self.org_y = None                            # [px] Origin of the UV frame in the y-direction
    self.sf_x = None                             # [m/pixel] Scaling factor from NED to UV frame in the x-direction
    self.sf_y = None                             # [m/pixel] Scaling factor from NED to UV frame in the y-direction
    self.x_cell_size = None                      # [m] Size of each cell in the x-direction storing height data
    self.y_cell_size = None                      # [m] Size of each cell in the y-direction storing height data
    self.sn_x_min = None                         # [px] Minimum x-value in the surface normal map
    self.sn_x_max = None                         # [px] Maximum x-value in the surface normal map
    self.sn_y_min = None                         # [px] Minimum y-value in the surface normal map
    self.sn_y_max = None                         # [px] Maximum y-value in the surface normal map
    self.num_sites = None                        # Number of landing sites associated with this packet
    self.center_coords_ned = np.array([])        # [m] NED coordinates of the landing site(s)
    self.center_coords_uv = np.array([])         # [px] UV coordinates of the landing site(s)
    self.radii_ned = np.array([])                # [m] Radii of the landing site(s)
    self.radii_uv = np.array([])                 # [px] Radii of the landing site(s)
    self.center_coords_ned_coarse = np.array([]) # [m] used for the global data packet to store the coarse landing region info
    self.center_coords_uv_coarse = np.array([])  # [m] used for the global data packet to store the coarse landing region info
    self.radii_ned_coarse = np.array([])         # [m] used for the global data packet to store the coarse landing region info
    self.radii_uv_coarse = np.array([])          # [m] used for the global data packet to store the coarse landing region info
    self.safety_map = np.array([])               # [px] Safety map (binary) of the surface normal map
    self.angle_map = np.array([])                # [deg] Elevation angle map of the surface normal map
    self.surf_norm = np.array([])                # [px] Surface normal map
    self.safety_map_circles = np.array([])       # [px] Safety map with landing site circles overlayed
    self.skeleton = np.array([])                 # [px] medial axis transform (MAT) skeleton of the safety map
    self.distance_map = np.array([])             # [m] Map of distances to the nearest unsafe pixel in the safety map

  def downsample_pointcloud(self):
    """
    Performs a downsampling operation on the point cloud "pcd_global"
    """
    pcd = self.pcd_global
    x_cell_size = self.x_cell_size
    y_cell_size = self.y_cell_size
    
    # Bin the x-y values
    pcd_bin = pcd.copy()
    pcd_bin[:,0] = np.round(0.5+pcd_bin[:,0]/x_cell_size)
    pcd_bin[:,1] = np.round(0.5+pcd_bin[:,1]/y_cell_size)
    
    # Remove non-unique elements in terms of z-values
    (pcd_uni,idx_uni) = np.unique(pcd_bin, axis=0, return_index=True)
    
    # Sort lexicographically (z-y-x order)
    idx_sort = np.lexsort(np.flipud(pcd_uni.T))
    pcd_sort = pcd_uni[idx_sort]
    
    # Find indices where the x-y (bin) values are same on both adjacent rows (means z-value is not extremal due to sorting)
    pcd_lshift = np.roll(pcd_sort,1,axis=0)
    pcd_rshift = np.roll(pcd_sort,-1,axis=0)
    idx_adj = np.where((pcd_lshift[:,0] == pcd_sort[:,0]) & (pcd_rshift[:,0] == pcd_sort[:,0]) & (pcd_lshift[:,1] == pcd_sort[:,1]) & (pcd_rshift[:,1] == pcd_sort[:,1]))[0]
    idx_rem = np.setdiff1d(np.arange(pcd_sort.shape[0]),idx_adj)
    pcd_rem = pcd_sort[idx_rem]
        
    # Culled idxs are a composition of all the index swaps and downsamples from previous steps
    idx_culled = idx_uni[idx_sort][idx_rem]
    self.pcd_culled = pcd[idx_culled]
    self.pcd_binned = pcd_bin[idx_culled]
    
    # # Inject a duplicate point at bins where only one point is present.
    # pcd_lshift = np.roll(pcd_rem,1,axis=0)
    # pcd_rshift = np.roll(pcd_rem,-1,axis=0)
    # idx_noadj = np.where((pcd_lshift[:,0] != pcd_rem[:,0]) & (pcd_rshift[:,0] != pcd_rem[:,0]) | (pcd_lshift[:,1] != pcd_rem[:,1]) & (pcd_rshift[:,1] != pcd_rem[:,1]))[0]
    
    # # Perform injection
    # self.pcd_culled = np.insert(self.pcd_culled, idx_noadj, self.pcd_culled[idx_noadj], axis=0)
    # self.pcd_binned = np.insert(self.pcd_binned, idx_noadj, self.pcd_binned[idx_noadj], axis=0)

  def pc_to_surf_normal(self, params):
    """
    Converts the culled point cloud to a surface normal image
    """
    model_x_data = np.linspace(min(self.pcd_culled[:,0]), max(self.pcd_culled[:,0]), params.grid_res)
    model_y_data = np.linspace(min(self.pcd_culled[:,1]), max(self.pcd_culled[:,1]), params.grid_res)
    X, Y = np.meshgrid(model_x_data, model_y_data)
    
    f_linear = interpolate.LinearNDInterpolator(list(zip(self.pcd_culled[:,0], self.pcd_culled[:,1])), self.pcd_culled[:,2])
    Z = f_linear(X, Y)
    
    surf_norm_img = surface_normal_from_interp_model(model_x_data, model_y_data, Z, params)

    self.surf_norm = cv2.cvtColor(surf_norm_img, cv2.COLOR_BGR2RGB)
  
  def mask_surf_norm(self, scale_factor = 2):
    """
    Masks out nonconvex interpolated regions of the surface normal map
    """
    surf_norm = self.surf_norm
    x_cell_size = self.x_cell_size
    y_cell_size = self.y_cell_size

    masked_surf_norm = np.zeros_like(surf_norm)

    self.find_NED_origin_uv()

    x_bins = (self.x_cell_size*self.pcd_binned[::2,0]).tolist()
    y_bins = (self.y_cell_size*self.pcd_binned[::2,1]).tolist()
    u_vec, v_vec = self.ned2uv(x_bins, y_bins)
        
    box_x_min, box_y_min = self.ned2uv(0, 0)
    box_x_max, box_y_max = self.ned2uv(x_cell_size/2, y_cell_size/2)

    cell_width  = (box_x_max - box_x_min)*scale_factor
    cell_height = (box_y_max - box_y_min)*scale_factor
    
    for idx in range(len(u_vec)):
      u = u_vec[idx] - (2*self.org_y - self.sn_x_max)
      v = v_vec[idx]
      v1 = int(0.5+(v - cell_height))
      u1 = int(0.5+(u - cell_width))
      v2 = int(0.5+(v + cell_height))
      u2 = int(0.5+(u + cell_width))
      masked_surf_norm = cv2.rectangle(masked_surf_norm, (v1, u1), (v2, u2), (255, 255, 255), -1)
    self.surf_norm = cv2.bitwise_and(surf_norm, surf_norm, mask = masked_surf_norm[:,:,0])
    self.surf_norm = np.flipud(self.surf_norm)
  
  def surf_norm_to_safety_map(self, params):
    """
    Generates the safety map by thresholding max elevation angle ("alpha")
    """
    flat_vec = np.array([1,0,0])
    norm_png = self.surf_norm.astype(np.float16)/255

    cos_theta_vec = flat_vec @ norm_png.reshape(norm_png.shape[0]*norm_png.shape[1],3).T
    cos_theta_mat = cos_theta_vec.reshape(norm_png.shape[0], norm_png.shape[1])
    theta_mat = np.arccos(cos_theta_mat)*180/np.pi
    safety_map = np.zeros_like(self.surf_norm[:,:,0])
    safety_map[theta_mat < params.alpha] = 255
    safety_map_color = np.zeros((params.grid_res,params.grid_res,3))
    safety_map_color[:,:,0] = safety_map
    safety_map_color[:,:,1] = safety_map
    safety_map_color[:,:,2] = safety_map
    self.safety_map = safety_map_color.astype(np.uint8)
    self.angle_map = theta_mat

  def find_landing_site(self):
      """
      Uses medial axis transform (MAT) skeletonization to compute the top-N largest inscribed circles in the safety map
      """
      # Compute the medial axis (skeleton) and the distance transform
      data = prep_safety_mask(self.safety_map)
      skel, distance = medial_axis(data, return_distance=True)
      self.distance = distance
      
      # Find the top-N largest inscribes circles in the skeleton
      center_coords, radii = topNCircles(distance, self.num_sites)
      u_vec = np.zeros(len(center_coords))
      v_vec = np.zeros(len(center_coords))
      for i in range(len(center_coords)):
          u_vec[i] = center_coords[i][0][0]
          v_vec[i] = center_coords[i][1][0]

      # Param update
      self.radii_uv = radii
      self.radii_ned = self.sf_x*np.array(radii).astype("float") # Convert new radii to numpy array
      self.center_coords_uv = np.array([u_vec, v_vec]).T
      for idx in range(self.num_sites):
        self.center_coords_uv_to_ned(idx)
  
  def update_landing_site(self, params):
    """
    Updates the landing site radius based on updated safety map
    """
    # Construct flipped (binary) safety map
    safety_map_bin = binarize(self.safety_map)
    safety_map_bin_cp = np.copy(safety_map_bin)
    
    # Apply a circular mask to the safety map
    region_radius = int(params.grid_res/2)
    mask = np.zeros((safety_map_bin_cp.shape[0], safety_map_bin_cp.shape[1]), dtype = "uint8")
    cv2.circle(mask, (int(self.safety_map.shape[0]/2), int(self.safety_map.shape[0]/2)), region_radius, 255, -1)
    masked_safety_map = cv2.bitwise_and(safety_map_bin_cp, safety_map_bin_cp, mask=mask)
    
    # Calculate geometric information about site
    max_x = int(self.safety_map.shape[0]/2 + region_radius)
    min_x = int(self.safety_map.shape[0]/2 - region_radius)
    max_y = int(self.safety_map.shape[1]/2 + region_radius)
    min_y = int(self.safety_map.shape[1]/2 - region_radius)
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    off_min_x = 0 if min_x < 0 else min_x
    off_min_y = 0 if min_y < 0 else min_y
    off_max_x = masked_safety_map.shape[0] if max_x > masked_safety_map.shape[0] else max_x
    off_max_y = masked_safety_map.shape[1] if max_y > masked_safety_map.shape[1] else max_y
    rel_center_x = center_x - off_min_x
    rel_center_y = center_y - off_min_y

    # Synthesize landing site and compute the new radius
    landing_site = masked_safety_map[off_min_x:off_max_x, off_min_y:off_max_y]
    landing_site = np.invert(landing_site.astype(np.uint8))
    unsafe_pixels = np.nonzero(landing_site)
    try:
      new_radius = np.sqrt((rel_center_x - unsafe_pixels[0])**2 + (rel_center_y - unsafe_pixels[1])**2).min()
    except ValueError:
      new_radius = region_radius

    # Param update
    if self.radii_uv[0] > 0:
      radius_sf = new_radius/self.radii_uv[0]
    else:
      radius_sf = 0
    self.radii_ned[0] = self.radii_ned[0]*radius_sf
    self.radii_uv[0] = new_radius
    return new_radius

  def scale_uv_2_world(self):
    self.sn_x_min = 0
    self.sn_x_max = self.surf_norm.shape[1]
    self.sn_y_min = 0
    self.sn_y_max = self.surf_norm.shape[0]
    self.pcd_x_min = self.pcd_global[:,0].min()
    self.pcd_x_max = self.pcd_global[:,0].max()
    self.pcd_y_min = self.pcd_global[:,1].min()
    self.pcd_y_max = self.pcd_global[:,1].max()
    self.sf_x = (self.pcd_x_max-self.pcd_x_min)/(self.sn_x_max - self.sn_x_min)
    self.sf_y = (self.pcd_y_max-self.pcd_y_min)/(self.sn_y_max - self.sn_y_min)
  
  def ned2uv(self, x, y):
    self.scale_uv_2_world()
    u = (y / self.sf_y) + self.org_y
    v = (x / self.sf_x) + self.org_x
    return u, v

  def uv2ned(self, u, v):
    del_x = v - self.org_x
    del_y = u - self.org_y
    x_scaled = del_x*self.sf_x
    y_scaled = del_y*self.sf_y
    pcd_kd_tree = scipy.spatial.KDTree(self.pcd_global[:,:2])
    dist,idx_tree = pcd_kd_tree.query(np.array([x_scaled, y_scaled]))
    if dist > 4:
      print("Warning! Your landing site is probably in a interpolated region where no points are present")
    x,y,z = self.pcd_global[idx_tree,:]
    return x,y,z

  def center_coords_uv_to_ned(self, idx):
    if self.center_coords_uv.size == 0:
      self.center_coords_uv = np.zeros((self.num_sites,2))
    if self.center_coords_ned.size == 0:
      self.center_coords_ned = np.zeros((self.num_sites,3))
    x,y,z = self.uv2ned(*self.center_coords_uv[idx].tolist())
    self.center_coords_ned[idx] = np.array([x,y,z])

  def center_coords_ned_to_uv(self, idx):
    if self.center_coords_uv.size == 0:
      self.center_coords_uv = np.zeros((self.num_sites,2))
    if self.center_coords_ned.size == 0:
      self.center_coords_ned = np.zeros((self.num_sites,3))
    u, v = self.ned2uv(*self.center_coords_ned[idx][:2].tolist())
    self.center_coords_uv[idx] = np.array([u,v])
    
  def find_NED_origin_uv(self):
    # Find NED Origin in UV Pixel Space
    self.scale_uv_2_world()
    x_pcd_to_surface = self.sn_x_min - (self.sn_x_max-self.sn_x_min)/(self.pcd_x_max- self.pcd_x_min) * (self.pcd_x_min)
    # y_pcd_to_surface = self.sn_y_min + (self.sn_y_max-self.sn_y_min) * (0 - self.pcd_y_min)/(self.pcd_y_max - self.pcd_y_min)
    y_pcd_to_surface = self.sn_y_min - (self.sn_y_max-self.sn_y_min)/(self.pcd_y_max- self.pcd_y_min) * (self.pcd_y_min)
  
    self.org_x = x_pcd_to_surface
    self.org_y = y_pcd_to_surface