from torch.utils.data import DataLoader
from dataloader import DenoiseDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_side_by_side(x,y):
  assert x.shape[0] == y.shape[0]
  batch_size = x.shape[0]
  fig, axs = plt.subplots(batch_size, 3)
  for i in range(batch_size):
    axs[i, 0].imshow(x[i,1:4].permute(1, 2, 0))
    min_val = torch.min(x[i,0])
    max_val = torch.max(x[i,0])
    axs[i, 1].imshow(x[i,0])
    axs[i, 2].imshow(np.squeeze(y[i]), vmin = min_val, vmax = max_val)
  plt.show()


dir = os.path.normpath(r"D:\Doktorat\Badania\DEM-waterlevel\dataset\train")
dataset = DenoiseDataset(dir, 256, augment=False)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)

x, y = next(iter(dataloader))

plot_side_by_side(x,y)
print(x)
