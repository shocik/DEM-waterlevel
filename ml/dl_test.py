from helper import denormalize
from torch.utils.data import DataLoader
from dataloader import DenoiseDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from helper import plot_side_by_side


dir = os.path.normpath(r"D:\Doktorat\Badania\DEM-waterlevel\dataset\train")
dataset = DenoiseDataset(dir, 256, augment=True, repeat=2, mode="level", return_names=True)
batch_size=8
dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

x, y, names = next(iter(dataloader))
x_dem = x[:,0].numpy()
x_ort = x[:,1:4].numpy()
y_dem = y[:,0].numpy()
for i in range(x_ort.shape[0]):
    x_dem[i] = denormalize(x_dem[i], "dem")
    x_ort[i] = denormalize(x_ort[i], "ort")
    y_dem[i] = denormalize(y_dem[i], "dem")
    print(f"{names[i]}; {y_dem[i]}")
[print(y) for y in y_dem]
plot_side_by_side(x_ort,x_dem)#,y_dem)
