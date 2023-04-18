from osgeo import gdal
import cv2
import os

PATH = r'C:\\Users\\Chris\\Downloads\\DTEEC_023957_1755_024023_1755_U01.img'
min = -4534.51
max = -3685.57

raster = gdal.Open(PATH)

array = raster.ReadAsArray()
array[array < min] = min
array[array > max] = max
norm_array = array - min
norm_array = norm_array/(norm_array.max())
norm_array = norm_array*65535
norm_array = norm_array.astype('uint16')
cv2.imwrite('msl_landing_site.png', norm_array)