import cv2
import numpy as np
# import matplotlib.pyplot as plt
from pdb import set_trace as debug

# def plot_coarse_images(halss_data):
#     plt.rcParams.update({'font.size': 6})

#     sn = cv2.cvtColor(halss_data.surf_norm, cv2.COLOR_BGR2RGB)
#     var = cv2.cvtColor(halss_data.variance_map_vis, cv2.COLOR_BGR2RGB)
#     skel = cv2.cvtColor(halss_data.skeleton, cv2.COLOR_BGR2RGB)
#     safety_map_circles = cv2.cvtColor(halss_data.safety_map_circles, cv2.COLOR_BGR2RGB)
    
#     fig, axs = plt.subplots(2,2, dpi=200)
#     fig.suptitle('Global Coarse Images')
#     axs[0,0].imshow(sn)
#     axs[0,0].set_title('Coarse Surface Normal')
#     axs[0,1].imshow(var)
#     axs[0,1].set_title("Variance Map")
#     axs[1,0].imshow(skel)
#     axs[1,0].set_title("Medial Axis Skeleton")
#     axs[1,1].imshow(safety_map_circles)
#     axs[1,1].set_title('Region Selection')
#     plt.show()
#     return

# def plot_fine_images(halss_data):
#     for idx, halss_local in enumerate(halss_data.halss_locals):
#         plt.figure()
#         # plt.rcParams.update({'font.size': 6})
#         sn = cv2.cvtColor(halss_local.surf_norm, cv2.COLOR_BGR2RGB)
#         skel = cv2.cvtColor(halss_local.skeleton, cv2.COLOR_BGR2RGB)
#         safety_map_circles = cv2.cvtColor(halss_local.safety_map_circles, cv2.COLOR_BGR2RGB)
        
#         fig, axs = plt.subplots(1, 3, dpi =200)
#         fig.suptitle('Local Fine Images for Region ' + str(idx+1))
#         axs[0].imshow(sn)
#         axs[0].set_title('Fine Surface Normal')
#         axs[1].imshow(skel)
#         axs[1].set_title("Medial Axis Skeleton")
#         axs[2].imshow(safety_map_circles)
#         axs[2].set_title('Region Selection')
#         plt.show()
#     return

def plotCircles(centers, radii, image, colors = None, thickness = 1, fill = True, border = True, center = True, fill_frac = 3):
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
        if colors is not None:
            if isinstance(colors[0], list) or isinstance(colors[0], tuple):
                color = colors[i]
            else:
                color = colors
        else:
            color = (0,0,255)
        color_bright = [255-(255-c)/fill_frac for c in color]
        
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

        # Plot circle on image such that any unsafe (black) spots in original image are not covered
        image_clone = color_img.copy()
        safe_idxs = np.nonzero(np.invert((image_clone[:,:,0] == 0) & (image_clone[:,:,1] == 0) & (image_clone[:,:,2] == 0)))
        if fill:
            color_img_new = cv2.circle(color_img.copy(), (yc,xc), radius, color_bright, cv2.FILLED)
            color_img[safe_idxs[0],safe_idxs[1],:] = color_img_new[safe_idxs[0],safe_idxs[1],:]
        if border:
            color_img_new = cv2.circle(color_img.copy(), (yc,xc), radius, color, thickness)
            color_img[safe_idxs[0],safe_idxs[1],:] = color_img_new[safe_idxs[0],safe_idxs[1],:]
        if center:
            color_img_new = cv2.circle(color_img.copy(), (yc,xc), 1, color, -1)
            color_img[safe_idxs[0],safe_idxs[1],:] = color_img_new[safe_idxs[0],safe_idxs[1],:]
            
    return color_img