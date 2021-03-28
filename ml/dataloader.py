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

class DenoiseDataset(Dataset):
    def __init__(
            self, 
            dir, 
            img_size,
            augment=False,
            normalize=True,
            names=[],
            repeat=1,
            return_names=False
    ):

        self.img_size = (img_size, img_size)
        self.augment = augment
        self.normalize = normalize
        self.x_dem_dir = os.path.join(dir,"x_dem")
        self.x_ort_dir = os.path.join(dir,"x_ort")
        self.y_dem_dir = os.path.join(dir,"y_dem")
        self.return_names = return_names
        if names:
            self.names = names*repeat
        else:
            self.names = os.listdir(self.x_ort_dir)*repeat
        print(self.names)
        self.x_dem_fps = [os.path.join(self.x_dem_dir, name) for name in self.names]
        self.x_ort_fps = [os.path.join(self.x_ort_dir, name) for name in self.names]
        self.y_dem_fps = [os.path.join(self.y_dem_dir, name) for name in self.names]
        
    def augmentation(self, x_dem, x_ort, y_dem, i):
        random.seed(i)
        rotation = random.randint(0,3)
        random.seed(i)
        flip_x = bool(random.getrandbits(1))
        random.seed(i+123456789)
        flip_y = bool(random.getrandbits(1))
        random.seed(i)
        offset = -random.uniform(0., 200.)
        
        #x_dem += offset
        #y_dem += offset

        x_ort = np.copy(np.rot90(x_ort,rotation,(2,1)))
        x_dem = np.copy(np.rot90(x_dem,rotation,(2,1)))
        y_dem = np.copy(np.rot90(y_dem,rotation,(2,1)))
        if flip_x:  
            y_dem = np.copy(np.flip(y_dem,1))
            x_ort = np.copy(np.flip(x_ort,1))
            x_dem = np.copy(np.flip(x_dem,1))
        if flip_y:
            y_dem = np.copy(np.flip(y_dem,2))
            x_ort = np.copy(np.flip(x_ort,2))
            x_dem = np.copy(np.flip(x_dem,2))
        return (x_dem, x_ort, y_dem)
   
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

        y_dem = np.load(self.y_dem_fps[i])
        y_dem = cv2.resize(y_dem,self.img_size)
        y_dem = np.expand_dims(y_dem,0)

        if self.augment:
            x_dem, x_ort, y_dem = self.augmentation(x_dem, x_ort, y_dem, i)
        if self.normalize:
            x_dem = normalize(x_dem, "dem")
            x_ort = normalize(x_ort, "ort")
            y_dem = normalize(y_dem, "dem")

        x = np.vstack((x_dem, x_ort))
        y = y_dem

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.return_names:
            return x, y, os.path.basename(self.x_dem_fps[i])
        else:
            return x, y
        
    def __len__(self):
        return len(self.names)