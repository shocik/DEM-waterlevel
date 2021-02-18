from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import cv2
import numpy as np
import torch

class DenoiseDataset(Dataset):
    def __init__(
            self, 
            dir, 
            img_size,
            count=-1
    ):
        self.count = count
        self.img_size = (img_size, img_size)
        self.x_dem_dir = os.path.join(dir,"x_dem")
        self.x_ort_dir = os.path.join(dir,"x_ort")
        self.y_dem_dir = os.path.join(dir,"y_dem")
        self.names = os.listdir(self.x_ort_dir)
        if count>0:
          self.names = self.names[:count]
        self.x_dem_fps = [os.path.join(self.x_dem_dir, name) for name in self.names]
        self.x_ort_fps = [os.path.join(self.x_ort_dir, name) for name in self.names]
        self.y_dem_fps = [os.path.join(self.y_dem_dir, name) for name in self.names]

    def __getitem__(self, i):

        x_dem = np.load(self.x_dem_fps[i])
        x_dem = cv2.resize(x_dem,self.img_size)
        x_dem = np.expand_dims(x_dem,0)

        x_ort = np.load(self.x_ort_fps[i]).astype(np.float32)
        x_ort = np.moveaxis(x_ort, 0, -1)
        x_ort = cv2.resize(x_ort,self.img_size)
        x_ort = np.moveaxis(x_ort, -1, 0)
        x_ort = x_ort/255

        y_dem = np.load(self.y_dem_fps[i])
        y_dem = cv2.resize(y_dem,self.img_size)
        y_dem = np.expand_dims(y_dem,0)

        x = np.vstack((x_dem, x_ort))
        y = y_dem
        #image = self.image_transform(image)
        #mask = mask.astype(np.float32)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        
        return x, y
        
    def __len__(self):
        return len(self.names)