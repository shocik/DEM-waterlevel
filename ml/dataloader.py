from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import cv2
import numpy as np
import torch
import random
from helper import normalize
import matplotlib.pyplot as plt
import csv

PERMUTATION = 16

class DenoiseDataset(Dataset):
    def __init__(
            self, 
            dir, 
            img_size,
            augment=False,
            normalize=True,
            names=[],
            return_names=False,
            mode = "dem"#"level"
    ):
        self.dir = dir
        self.img_size = (img_size, img_size)
        self.augment = augment
        self.normalize = normalize
        self.x_dem_dir = os.path.join(dir,"x_dem")
        self.x_ort_dir = os.path.join(dir,"x_ort")
        self.y_dem_dir = os.path.join(dir,"y_dem")
        self.return_names = return_names
        self.mode = mode
        
        if mode=="level":
            self.level_dict = self.level_reader()
        if names:
            self.names = names
        else:
            if mode=="dem":
                self.names = os.listdir(self.x_ort_dir)
            elif mode=="level":
                self.names = list(self.level_dict.keys())
        print(self.names)
        self.x_dem_fps = [os.path.join(self.x_dem_dir, name) for name in self.names]
        self.x_ort_fps = [os.path.join(self.x_ort_dir, name) for name in self.names]
        if mode=="dem":
            self.y_dem_fps = [os.path.join(self.y_dem_dir, name) for name in self.names]

    def level_reader(self):
        level_dict = dict()
        #water_bool_dict = dict()
        with open(os.path.join(self.dir,'levels.csv'), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                level_dict[row[0]] = float(row[1])
        return level_dict

    def augmentation(self, x_dem, x_ort, y, i):
        m16 = i%16
        m4 = m16%4
        rotation = m16/4    #values from 0 to 3
        flip_x = m4 in [1,3]#
        flip_y = m4 in [2,3]# 0 - no flip, 1 - only flip x, 2 - only flip y, 3 - flip x and y.
        print(f"{i}, {rotation}, {flip_x}, {flip_y}")
        if rotation != 0:
            x_ort = np.copy(np.rot90(x_ort,rotation,(2,1)))
            x_dem = np.copy(np.rot90(x_dem,rotation,(2,1)))
            if self.mode == "dem":
                y = np.copy(np.rot90(y,rotation,(2,1)))
        if flip_x:
            x_ort = np.copy(np.flip(x_ort,1))
            x_dem = np.copy(np.flip(x_dem,1))
            if self.mode == "dem":
                y = np.copy(np.flip(y,1))
        if flip_y:
            x_ort = np.copy(np.flip(x_ort,2))
            x_dem = np.copy(np.flip(x_dem,2))
            if self.mode == "dem":
                y = np.copy(np.flip(y,2))
        return (x_dem, x_ort, y)
   
    def __getitem__(self, i):
        img_i = i//16
        x_dem = np.load(self.x_dem_fps[img_i])
        x_dem = cv2.resize(x_dem,self.img_size)
        x_dem = np.expand_dims(x_dem,0)
        x_ort = np.load(self.x_ort_fps[img_i]).astype(np.float32)
        x_ort = x_ort/255
        #(C,H,W)
        x_ort = np.moveaxis(x_ort, 0, -1)
        #(H,W,C)
        #plt.imshow(x_ort)
        #plt.show()
        x_ort = cv2.resize(x_ort,self.img_size)
        x_ort = np.moveaxis(x_ort, -1, 0)
        #(C,H,W)
        if self.mode=="dem":
            y = np.load(self.y_dem_fps[img_i])
            y = cv2.resize(y,self.img_size)
            y = np.expand_dims(y,0)
        elif self.mode=="level":
            y = np.array([self.level_dict[self.names[img_i]]]).astype(np.float32)
        if self.augment:
            x_dem, x_ort, y = self.augmentation(x_dem, x_ort, y, i)
        if self.normalize:
            x_dem = normalize(x_dem, "dem")
            x_ort = normalize(x_ort, "ort")
            y = normalize(y, "dem")

        x = np.vstack((x_dem, x_ort))

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.return_names:
            return x, y, os.path.basename(self.x_dem_fps[img_i])
        else:
            return x, y
        
    def __len__(self):
        length = len(self.names)
        if self.augment:
            length = length*PERMUTATION
        return length