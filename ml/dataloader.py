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

class DenoiseDataset(Dataset):
    def __init__(
            self, 
            dir, 
            img_size,
            augment=False,
            normalize=True,
            names=[],
            repeat=1,
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
        if names:
            self.names = names*repeat
        else:
            self.names = os.listdir(self.x_ort_dir)*repeat
        print(self.names)
        self.x_dem_fps = [os.path.join(self.x_dem_dir, name) for name in self.names]
        self.x_ort_fps = [os.path.join(self.x_ort_dir, name) for name in self.names]
        if mode=="dem":
            self.y_dem_fps = [os.path.join(self.y_dem_dir, name) for name in self.names]
        elif mode=="level":
            self.water_bool_dict, self.level_dict = self.level_reader()

    def level_reader(self):
        level_dict = dict()
        water_bool_dict = dict()
        with open(os.path.join(self.dir,'levels.csv'), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                water_bool_dict[row[0]] = int(row[1])
                level_dict[row[0]] = float(row[2])
        return water_bool_dict, level_dict

    def augmentation(self, x_dem, x_ort, y, i):

        random.seed(i)
        rotation = random.randint(0,3)
        random.seed(i)
        flip_x = bool(random.getrandbits(1))
        random.seed(i+123456789)
        flip_y = bool(random.getrandbits(1))
        random.seed(i)
        offset = random.uniform(-10., 10.)
        x_dem += offset
        y += offset

        x_ort = np.copy(np.rot90(x_ort,rotation,(2,1)))
        x_dem = np.copy(np.rot90(x_dem,rotation,(2,1)))
        if self.mode=="dem":
            y = np.copy(np.rot90(y,rotation,(2,1)))
        if flip_x:
            if self.mode=="dem":
                y = np.copy(np.flip(y,1))
            x_ort = np.copy(np.flip(x_ort,1))
            x_dem = np.copy(np.flip(x_dem,1))
        if flip_y:
            if self.mode=="dem":
                y = np.copy(np.flip(y,2))
            x_ort = np.copy(np.flip(x_ort,2))
            x_dem = np.copy(np.flip(x_dem,2))
        return (x_dem, x_ort, y)
   
    def __getitem__(self, i):

        x_dem = np.load(self.x_dem_fps[i])
        x_dem = cv2.resize(x_dem,self.img_size)
        x_dem = np.expand_dims(x_dem,0)
        
        x_ort = np.load(self.x_ort_fps[i]).astype(np.float32)
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
            y = np.load(self.y_dem_fps[i])
            y = cv2.resize(y,self.img_size)
            y = np.expand_dims(y,0)
        elif self.mode=="level":
            y = self.level_dict[self.names[i]]
        if self.augment:
            x_dem, x_ort, y = self.augmentation(x_dem, x_ort, y, i)
        if self.normalize:
            x_dem = normalize(x_dem, "dem")
            x_ort = normalize(x_ort, "ort")
            y = normalize(y, "dem")

        x = np.vstack((x_dem, x_ort))
        if self.mode=="level":
            y = np.array([ float(self.water_bool_dict[self.names[i]]), y ])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.return_names:
            return x, y, os.path.basename(self.x_dem_fps[i])
        else:
            return x, y
        
    def __len__(self):
        return len(self.names)