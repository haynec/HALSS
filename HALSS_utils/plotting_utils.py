import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_coarse_images(halss_data):
    plt.rcParams.update({'font.size': 6})

    sn = cv2.cvtColor(halss_data.surf_norm, cv2.COLOR_BGR2RGB)
    var = cv2.cvtColor(halss_data.variance_map_vis, cv2.COLOR_BGR2RGB)
    skel = cv2.cvtColor(halss_data.skeleton, cv2.COLOR_BGR2RGB)
    safety_map_circles = cv2.cvtColor(halss_data.safety_map_circles, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(2,2, dpi=200)
    fig.suptitle('Global Coarse Images')
    axs[0,0].imshow(sn)
    axs[0,0].set_title('Coarse Surface Normal')
    axs[0,1].imshow(var)
    axs[0,1].set_title("Variance Map")
    axs[1,0].imshow(skel)
    axs[1,0].set_title("Medial Axis Skeleton")
    axs[1,1].imshow(safety_map_circles)
    axs[1,1].set_title('Region Selection')
    plt.show()
    return