import torch
import numpy as np

# helper function to plot x, ground truth and predict images in grid
import matplotlib.pyplot as plt
def plot_side_by_side(x,y_dem_gt, y_dem_pr=None):
  batch_size = x.shape[0]
  fig, axs = plt.subplots(batch_size+1, 3 if y_dem_pr is None else 4)
  #if batch_size==1:
  #  axs = np.expand_dims(axs,0)
  for i in range(batch_size):
    axs[i, 0].imshow(x[i,1:4].permute(1, 2, 0))
    min_val = torch.min(x[i,0])
    max_val = torch.max(x[i,0])
    axs[i, 1].imshow(x[i,0], vmin = min_val, vmax = max_val)
    axs[i, 2].imshow(np.squeeze(y_dem_gt[i]), vmin = min_val, vmax = max_val)
    if y_dem_pr is not None:
        axs[i, 3].imshow(np.squeeze(y_dem_pr[i]), vmin = min_val, vmax = max_val)
    #axs = np.squeeze(axs)
  plt.show()