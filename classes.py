# Import the repository root directory (one level above the ros root directory)
import sys,os
root_path = os.path.abspath(os.getcwd() + "/..")
sys.path.append(root_path)

# Base imports
import cv2
import numpy as np
from scipy import interpolate
import scipy.ndimage
import skimage
from skimage.morphology import medial_axis

# Custom imports
from HALSS.utils.utils import *

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
    self.x_cell_size_coarse = None
    self.y_cell_size_coarse = None
    self.x_cell_size_fine = None
    self.y_cell_size_fine = None
    self.alpha = None
    self.max_sites = None
    self.media_save_path = None
    self.lambdas = None

class halss_data_packet:
  def __init__(self):
    self.pcd_global = np.array([])
    self.pcd_local = np.array([])
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
    self.halss_locals = []

  def scale_uv_2_world(self):
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

  def uv2ned(self, u, v, flags):
    del_x = v - self.org_x
    del_y = u - self.org_y
    x_ned_scaled = del_x*self.sf_x
    y_ned_scaled = del_y*self.sf_y

    pcd_kd_tree = scipy.spatial.KDTree(self.pcd_global[:,:2])
    dist, idx = pcd_kd_tree.query(np.array([x_ned_scaled, -y_ned_scaled]))
    if dist > 4:
      print("Warning! Your landing site is probably in a interpolated region where no point could points are present")
    x_ned_actual, y_ned_actual, z_ned_actual = self.pcd_global[idx,:]

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
    pcd = self.pcd_global
    x_cell_size = self.x_cell_size
    y_cell_size = self.y_cell_size

    for pt in pcd:
        x = int(0.5+pt[0]/x_cell_size)
        y = int(0.5+pt[1]/y_cell_size)
        if not x in pcd_culled_dict:
            pcd_culled_dict[x] = {}
        col = pcd_culled_dict[x]
        if not y in col:
            col[y] = [pt, pt]
        else:
            if pt[2] < col[y][0][2]:
                col[y][0] = pt
            elif pt[2] > col[y][1][2]:
                col[y][1] = pt
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

  def find_NED_origin_uv(self):
    # Find NED Origin in UV Pixel Space
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
  
  def surf_norm_thresh(self, params):
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

    self.find_NED_origin_uv()

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
      kernel = np.ones((2, 2), np.uint8)
      skel = cv2.dilate(skel.astype(np.uint8), kernel, iterations=1)
      dist_on_skel = distance * skel

      center_coords, radii = topNCircles(distance, self.num_circles)
      u_vec = np.zeros(len(center_coords))
      v_vec = np.zeros(len(center_coords))
      for i in range(len(center_coords)):
          u_vec[i] = center_coords[i][0][0]
          v_vec[i] = center_coords[i][1][0]

      circles = plotCircles((u_vec, v_vec), radii, data)
      self.radii_uv = radii
      skeleton = dist_on_skel[2:dist_on_skel.shape[0]-2, 2:dist_on_skel.shape[1]-2]
      skel_color = cv2.applyColorMap((normalize(skeleton)*255).astype(np.uint8), cv2.COLORMAP_HOT)
      contours, hierarchy = cv2.findContours(self.safety_map.astype(np.uint8)[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      self.skeleton = cv2.drawContours(skel_color, contours, -1, (255, 255, 255), 1)

      self.safety_map_circles = circles
      self.distance = distance
      for idx, _ in enumerate(center_coords):
        self.uv2ned(center_coords[idx][0][0], center_coords[idx][1][0], flags)
  
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