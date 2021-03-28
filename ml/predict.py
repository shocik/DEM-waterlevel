# -*- coding: utf-8 -*-

import os
import sys

#set workdir
#os.chdir("/content/drive/MyDrive/DEM-waterlevel/ml/")

#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader import DenoiseDataset
from torchinfo import summary
import time
import copy
import pdb
from tqdm import tqdm
from helper import plot_side_by_side, denormalize

#training parameters in neptune format
PARAMS = {

    "img_size": 256,
    "model": "vgg_unet",
    "batch_size": 16,
    'epochs': 1000,
    'patience': 10,
    'image_preload': False,
    'task': "predict",
    'min_improvement': 0.001,
    'neptune': False
}
if len(sys.argv)>1:
  assert sys.argv[1] in ["train", "predict", "all"]
  PARAMS["task"]=sys.argv[1]
print(PARAMS)
#model loading
if PARAMS['model'] == "vgg_unet":
  from models.vgg_unet import VggUnet
  model = VggUnet()
  model_src = 'vgg_unet.py'
elif PARAMS['model'] == "autoencoder":
  from models.autoencoder import Autoencoder
  model = Autoencoder(PARAMS['img_size'])
  model_src = 'autoencoder.py'


#dataset configuration
dataset_dir = os.path.normpath("dataset")
train_dir = os.path.join(dataset_dir,"train")
test_dir = os.path.join(dataset_dir,"test")

batch_size = PARAMS['batch_size']

#model structure preview
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model_stats = summary(model, input_size=(PARAMS['batch_size'], 4, PARAMS['img_size'], PARAMS['img_size']))

# load weights
model.load_state_dict(torch.load("state_dict.pth", map_location="cpu"))
device = torch.device('cpu')
model = model.to(device)
# denormalization function


# visualize example segmentation
import math
model.eval()   # Set model to evaluate mode
test_dataset = DenoiseDataset(test_dir, img_size=PARAMS['img_size'], return_names=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

predictions_pth = "predictions"

for inputs, gts, names in tqdm(test_loader):
    inputs = inputs.to(device)
    gts = gts.to(device)

    gts = gts.data.cpu()
    pred = model(inputs)

    pred = pred.data.cpu()
    inputs = inputs.data.cpu()

    # use helper function to plot
    x_dem = inputs[:,0].numpy()
    x_ort = inputs[:,1:4].numpy()
    y_dem_gt = gts[:,0].numpy()
    y_dem_pr = pred[:,0].numpy()
    for i in range(x_ort.shape[0]):
        x_dem[i] = denormalize(x_dem[i], "dem")
        x_ort[i] = denormalize(x_ort[i], "ort")
        y_dem_gt[i] = denormalize(y_dem_gt[i], "dem")
        y_dem_pr[i] = denormalize(y_dem_pr[i], "dem")
        np.save(os.path.join(predictions_pth,names[i]),y_dem_pr[i])
    fig = plot_side_by_side(x_ort, x_dem, y_dem_gt, y_dem_pr)
