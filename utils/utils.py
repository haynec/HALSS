import numpy as np
import cv2
from skimage.morphology import medial_axis

def invert(png):
    # Invert the image
    inverted = 255 - png
    return inverted

def scale_image(img, pcd_combined_array):
    # Takes in image and {min_x, max_x, min_y, max_y} of point cloud and scales the image proportionally
    resize_fac = ((pcd_combined_array[:,1].max()-pcd_combined_array[:,1].min())/(pcd_combined_array[:,0].max()-pcd_combined_array[:,0].min()))
    img = cv2.resize(img, (img.shape[0], int(img.shape[1]*resize_fac)))
    return img

def scale_theta(img):
    # 
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

def maximum_possible_points(pcd, x_cell_size, y_cell_size):
    x_bins = np.arange(pcd[:,0].min(), pcd[:,0].max(), x_cell_size)
    y_bins = np.arange(pcd[:,1].min(), pcd[:,1].max(), y_cell_size)
    return len(x_bins)*len(y_bins)*2

def normalize(img):
    X = img.copy()
    xmin = np.amin(X)
    xmax = np.amax(X)
    X = (X - xmin) / (xmax - xmin)
    return X.astype(np.float32)

def getRotMatrix_uv2worldNED(client):
    cam_info = client.simGetCameraInfo(3)
    kin = client.simGetGroundTruthKinematics()
    orientation = kin.orientation
    q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                    [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                    [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
    return rotation_matrix, kin, cam_info

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
    elif type(centers[0]) is float:
        idx = 1
    else:
        idx = len(centers[0])
    
    if type(radii) is int: # Captures the case of only one circle
        radii = [radii]
    elif type(radii) is float:
        radii = [radii]
    else:
        radii = radii
        
    for i in range(idx):
        radius = int(radii[i])
        if type(centers[0]) is int:
            xc = int(centers[0]+2)
            yc = int(centers[1]+2)
        elif type(centers[0]) is float:
            xc = int(centers[0]+2)
            yc = int(centers[1]+2)
        else:
            xc = int(centers[0][i]+2)
            yc = int(centers[1][i]+2)
        # color_img = cv2.circle(color_img, (yc,xc), radius, color_fill, cv2.FILLED)
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

def remove_pixels_around_border(mask, border_size = 2):
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