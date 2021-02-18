from torch.utils.data import DataLoader
from dataloader import DenoiseDataset
import os

dir = os.path.normpath(r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\dataset\train")
dataset = DenoiseDataset(dir, 256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

x, y = next(iter(dataloader))
import matplotlib.pyplot as plt; import numpy as np; plt.imshow(np.moveaxis(x[0].numpy()[1:,:,:], 0, -1)); plt.show()
print(x)
