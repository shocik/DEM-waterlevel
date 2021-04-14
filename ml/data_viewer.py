import os
import numpy as np
import torch
from helper import plot_side_by_side
import cv2

dir = "dataset/train"
x_dem_dir = os.path.join(dir,"x_dem")
x_ort_dir = os.path.join(dir,"x_ort")
y_dem_dir = os.path.join(dir,"y_dem")
names = ["97.npy", "98.npy"]#os.listdir(x_ort_dir)

x_dem_fps = [os.path.join(x_dem_dir, name) for name in names]
x_ort_fps = [os.path.join(x_ort_dir, name) for name in names]
y_dem_fps = [os.path.join(y_dem_dir, name) for name in names]
img_size = (256,256)
for i in range(len(x_ort_fps)):
    x_dem = np.load(x_dem_fps[i])
    x_dem = cv2.resize(x_dem,img_size)
    x_dem = np.expand_dims(x_dem,0)


    x_ort = np.load(x_ort_fps[i]).astype(np.float32)
    x_ort = np.moveaxis(x_ort, 0, -1)
    x_ort = cv2.resize(x_ort,img_size)
    x_ort = np.moveaxis(x_ort, -1, 0)
    x_ort = x_ort/255
    x_ort = np.expand_dims(x_ort,0)

    y_dem = np.load(y_dem_fps[i])
    y_dem = cv2.resize(y_dem,img_size)
    y_dem = np.expand_dims(y_dem,0)

    #x = np.vstack((x_dem, x_ort))
    #y = y_dem
    #image = image_transform(image)
    #mask = mask.astype(np.float32)
    print(x_ort_fps[i])
    plot_side_by_side(x_ort,x_dem,y_dem)
