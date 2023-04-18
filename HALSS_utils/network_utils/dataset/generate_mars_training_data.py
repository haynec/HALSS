import airsim
import sys
import numpy as np
import os
import tempfile
import pprint
import cv2

def getCombinedMask(surfNorm, seg):
 
    surfaceNormal_vector_form = surfNorm.copy()

    # conversion from surface normal to pointing vector: color value / 255 * 2.0 - 1.0
    w,h,channels = surfaceNormal_vector_form.shape
    for i in range(channels):
        for j in range(w):
            for k in range(h):
                color_value = surfNorm[j,k,i]
                temp = int(color_value / 255 * 2.0 - 1.0)
                surfaceNormal_vector_form[j,k,i] = temp*255

    # let's get rid of the x and y components, we only care about z component
    surfaceNormal_vector_form[:,:,1] = 0
    surfaceNormal_vector_form[:,:,2] = 0

    # filter out small noise
    filtered = cv2.bilateralFilter(surfaceNormal_vector_form,30,150,150)
  
    # blur to get get smoother shapes without gaps
    blurred = cv2.GaussianBlur(filtered,(5,5),5)
    
    # blurred = cv2.GaussianBlur(surfaceNormal_vector_form,(5,5),5)

    # convert to gray for binairzation and thresholding
    gray_mask = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    b, surfaceNormal_mask = cv2.threshold(gray_mask,0,256,cv2.THRESH_BINARY)

    gray_mask = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    b, segmentation_mask = cv2.threshold(gray_mask,0,256,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_and(surfaceNormal_mask, segmentation_mask)
    return combined_mask, surfaceNormal_mask, segmentation_mask

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()

# airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()



# airsim.wait_key('Press any key to start path')
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

z_floor = -50
variance = 15^2
mean = 25
duration_low = 1
duration_hi = 6
yaw_values = np.array([0.0, 180.0, 90.0, 270.0])
for idx in range(300):
    
    filename = os.path.join(tmp_dir, str(idx))
    vx = np.random.normal(0, variance)
    vy = np.random.normal(0, variance)
    z = min(z_floor - np.random.normal(mean, variance), z_floor)
    duration = np.random.uniform(duration_low, duration_hi)
    # vx = 6
    # vy = 6
    # z = min(10*np.random.normal(mean, variance), z_floor)
    yaw = yaw_values[np.mod(idx,4)]
    print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=" + str(yaw) + " z value is " + str(z) + " iteration number " +str(idx))
    
    client.moveByVelocityZAsync(vx,vy,z,duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, yaw)).join()
    client.moveToZAsync(z, 10).join()
    client.hoverAsync().join()

    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        print(collision_info)
    #     client.reset()
    #     client.armDisarm(True)
    #     client.takeoffAsync().join()
    #     client.moveByVelocityZAsync(vx,vy,z,duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, yaw)).join()

    client.simPause(True)
    rawSegmentation = client.simGetImage("3", airsim.ImageType.Segmentation)
    rawSurfaceNormal = client.simGetImage("3", airsim.ImageType.SurfaceNormals)
    rawdepthPerspective = client.simGetImage("3", airsim.ImageType.DepthPerspective)
    rawScene = client.simGetImage("3", airsim.ImageType.Scene)
    if (rawSegmentation == None):
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        png_segmentation = cv2.imdecode(airsim.string_to_uint8_array(rawSegmentation), cv2.IMREAD_UNCHANGED)
        png_surface = cv2.imdecode(airsim.string_to_uint8_array(rawSurfaceNormal), cv2.IMREAD_UNCHANGED)
        png_depth = cv2.imdecode(airsim.string_to_uint8_array(rawdepthPerspective), cv2.IMREAD_UNCHANGED)
        png_scene = cv2.imdecode(airsim.string_to_uint8_array(rawScene), cv2.IMREAD_UNCHANGED)
        
        cv2.imwrite(os.path.normpath(filename + '_surfaceNormal.png'), png_surface)
        cv2.imwrite(os.path.normpath(filename + '_segmentation.png'), png_segmentation)

        combo, surf_mask, seg_mask =  getCombinedMask(cv2.imread(os.path.normpath(filename + '_surfaceNormal.png')), cv2.imread(os.path.normpath(filename + '_segmentation.png')))

        cv2.imwrite(os.path.normpath(filename + '_maskedSegmentation.png'), seg_mask)
        cv2.imwrite(os.path.normpath(filename + '_maskedSurfaceNormal.png'), surf_mask)
        cv2.imwrite(os.path.normpath(filename + '_combinedMask.png'), combo)
        cv2.imwrite(os.path.normpath(filename + '_scene.png'), png_scene)
        cv2.imwrite(os.path.normpath(filename + '_depthPerspective.png'), png_depth)
    client.simPause(False)    

airsim.wait_key('Press any key to reset to original state')
client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)    






