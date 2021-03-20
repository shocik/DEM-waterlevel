import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize(arr, type):
  if type=="dem":
      mu = 213.30667114257812
      sigma = 0.6645699739456177
      return (arr-mu)/sigma
  elif type=="ort":
      mu=[0.485, 0.456, 0.406]
      sigma=[0.229, 0.224, 0.225]
      for i in range(3):
          arr[i]=(arr[i]-mu[i])/sigma[i]
      return arr

def denormalize(arr, type):
  if type=="dem":
      mu = 213.30667114257812
      sigma = 0.6645699739456177
      return sigma*arr+mu
  elif type=="ort":
      mu=[0.485, 0.456, 0.406]
      sigma=[0.229, 0.224, 0.225]
      for i in range(3):
          arr[i]=sigma[i]*arr[i]+mu[i]
      return arr

# helper function to plot x, ground truth and predict images in grid
def plot_side_by_side(*arg):

    y_size = arg[0].shape[0]
    x_size = len(arg)
    fig, axs = plt.subplots(y_size, x_size)
    if y_size==1:
        axs = np.expand_dims(axs,0)
    for y in range(y_size):
        for x in range(x_size):
            img = arg[x][y]
            if len(img.shape)==3:
                img = np.moveaxis(img, 0, -1)
            axs[y, x].imshow(img)
            #x_ort = x[i,1:4]#.permute(1, 2, 0)
            #axs[i, 0].imshow(x_ort)

            #x_dem = x[i,0]
            #min_val = torch.min(x_dem)
            #max_val = torch.max(x_dem)
            #axs[i, 1].imshow(x_dem, vmin = min_val, vmax = max_val)
            #axs[i, 2].imshow(np.squeeze(y_dem_gt[i]), vmin = min_val, vmax = max_val)
            #if y_dem_pr is not None:
            #    axs[i, 3].imshow(denormalize(np.squeeze(y_dem_pr[i]),"dem"), vmin = min_val, vmax = max_val)
    axs = np.squeeze(axs)
    plt.show()
    return fig

