"""
import matplotlib.pyplot as plt
class LineDrawer(object):
    lines = []
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y)
        ax.figure.canvas.draw()

        self.lines.append(line)
ld = LineDrawer()
ld.draw_line()
"""

import os
import numpy as np
import torch
from helper import plot_side_by_side
import cv2
import matplotlib.pyplot as plt

dir = "dataset/test"
x_dem_dir = os.path.join(dir,"x_dem")
x_ort_dir = os.path.join(dir,"x_ort")
y_dem_dir_gt = os.path.join(dir,"y_dem")
y_dem_dir_pr = "predictions"
names = os.listdir(x_ort_dir)

x_dem_fps = [os.path.join(x_dem_dir, name) for name in names]
x_ort_fps = [os.path.join(x_ort_dir, name) for name in names]
y_dem_fps_gt = [os.path.join(y_dem_dir_gt, name) for name in names]
y_dem_fps_pr = [os.path.join(y_dem_dir_pr, name) for name in names]
img_size = (256,256)
for i in range(len(x_ort_fps)):
    x_dem = np.load(x_dem_fps[i])
    x_dem = cv2.resize(x_dem,img_size)


    x_ort = np.load(x_ort_fps[i]).astype(np.float32)
    x_ort = np.moveaxis(x_ort, 0, -1)
    x_ort = cv2.resize(x_ort,img_size)
    x_ort = np.moveaxis(x_ort, -1, 0)
    x_ort = x_ort/255

    y_dem_gt = np.load(y_dem_fps_gt[i])
    y_dem_gt = cv2.resize(y_dem_gt,img_size)

    y_dem_pr = np.load(y_dem_fps_pr[i])
    y_dem_pr = cv2.resize(y_dem_pr,img_size)
    #x = np.vstack((x_dem, x_ort))
    #y = y_dem_gt
    #image = image_transform(image)
    #mask = mask.astype(np.float32)
    print(x_ort_fps[i])

    fig, axs = plt.subplots(2, 2)
    min_val = np.amin(x_dem)
    max_val = np.amax(x_dem)
    x_ort = np.moveaxis(x_ort, 0, -1)
    axs[0, 0].imshow(x_ort)
    axs[0, 0].axis('off')
    axs[0, 0].title.set_text("Orthophoto")
    axs[0, 1].imshow(x_dem, vmin = min_val, vmax = max_val, cmap="jet")
    axs[0, 1].axis('off')
    axs[0, 1].title.set_text("Input DEM")
    axs[1, 0].imshow(y_dem_gt, vmin = min_val, vmax = max_val, cmap="jet")
    axs[1, 0].axis('off')
    axs[1, 0].title.set_text("Ground truth DEM")
    axs[1, 1].imshow(y_dem_pr, vmin = min_val, vmax = max_val, cmap="jet")
    axs[1, 1].axis('off')
    axs[1, 1].title.set_text("Deep learning corrected DEM")
    xy = None
    xy = plt.ginput(2)
    if xy:
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        for i in range(2):
            for j in range(2):
                axs[i,j].axis('off')
                axs[i,j].plot(x,y,"r")
        length = int(np.hypot(x[1]-x[0], y[1]-y[0]))
        x, y = np.linspace(x[0], x[1], length), np.linspace(y[0], y[1], length)
        


        plt.show()
        x_data = np.array([ element*10/256 for element in range(length) ])
        x_dem_zi = x_dem[x.astype(np.int), y.astype(np.int)]
        y_dem_gt_zi = y_dem_gt[x.astype(np.int), y.astype(np.int)]
        y_dem_pr_zi = y_dem_pr[x.astype(np.int), y.astype(np.int)]
        plt.plot(x_dem_zi, label="Input DEM")
        plt.plot(y_dem_gt_zi, label="Ground truth DEM")
        plt.plot(y_dem_pr_zi, label="Deep learning corrected DEM")
        plt.margins(x=0)
        plt.ylabel("Elevation (masl)")
        plt.xlabel("Distance (m)")
        plt.legend(loc="upper right")
        plt.show()
    #plot_side_by_side(x_ort,x_dem,y_dem_gt,y_dem_pr)
