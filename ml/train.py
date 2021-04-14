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
    "learning_rate": 0.01,
    "batch_size": 16,
    'epochs': 1000,
    'patience': 10,
    'image_preload': False,
    'task': "predict",
    'min_improvement': 0.001,
    'neptune': False,
    'mode': "level",
    'level_mode_loss_ratio': 0.5,
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

if PARAMS["neptune"]:
  #neptune initialization
 
  import configparser
  config = configparser.ConfigParser()
  config.read("./ml/config.cfg")
  import neptune
  neptune.init(project_qualified_name=config["neptune"]["project"],
              api_token=config["neptune"]["token"],
              )
  neptune.create_experiment(params=PARAMS, upload_source_files=['ml/overtrain.py', "ml/models/"+model_src])

#dataset configuration
dataset_dir = os.path.normpath("dataset")
train_dir = os.path.join(dataset_dir,"train")
test_dir = os.path.join(dataset_dir,"test")

train_set = DenoiseDataset(train_dir, img_size=PARAMS['img_size'], augment="True", repeat=20, mode=PARAMS["mode"])
test_set = DenoiseDataset(test_dir, img_size=PARAMS['img_size'])

batch_size = PARAMS['batch_size']
dataloaders = {
    'train': DataLoader(train_set, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0),
    'val': DataLoader(test_set, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0)
}

# load images - useful if you want to save some time by preloading images (very time-consuming) when 
# the model is still not fuctional and cant run standard training.
if PARAMS['image_preload']:
  for phase in dataloaders:
    for inputs, labels in tqdm(dataloader[phase]):
      pass



#model structure preview
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model_stats = summary(model, input_size=(PARAMS['batch_size'], 4, PARAMS['img_size'], PARAMS['img_size']))
if PARAMS["neptune"]:
  for line in str(model_stats).splitlines():
    neptune.log_text('model_summary', line)

if PARAMS["task"] in ["train", "all"]:
  from collections import defaultdict
  import torch.nn.functional as F
  def calc_loss(pred, target, metrics):
    if PARAMS["mode"]=="dem":
      loss = F.mse_loss(pred, target)
    elif PARAMS["mode"]=="level":
      classification_loss = F.binary_cross_entropy(pred[0], target[0])
      regression_loss = F.mse_loss(pred[1], target[1])
      loss = PARAMS['level_mode_loss_ratio']*classification_loss+(1-PARAMS['level_mode_loss_ratio'])*regression_loss
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
    elif

  def print_metrics(metrics, epoch_samples, phase):   
    print(epoch_samples) 
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        neptune.log_metric(phase+"_"+k, metrics[k] / epoch_samples) #log
    print("{}: {}".format(phase, ", ".join(outputs)))

  #training loop
  def train_model(model, dataloaders, optimizer, device, num_epochs=25, patience=-1):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improvement = 0
    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)
      
      since = time.time()

      # Each epoch has a training and validation phase

      for phase in ['train', 'val']:
        if phase == 'train':
          for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
            
          model.train()  # Set model to training mode
        else:
          model.eval()

        metrics = defaultdict(float)
        epoch_samples = 0
            
        for inputs, labels in tqdm(dataloaders[phase]):
          inputs = inputs.to(device)
          labels = labels.to(device)             

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss = calc_loss(outputs, labels, metrics)
            #print(model.encoder[0].weight.grad)
            # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              #pdb.set_trace()
              optimizer.step()

          # statistics
          epoch_samples += inputs.size(0)

        print_metrics(metrics, epoch_samples, phase)
        epoch_loss = metrics['loss'] / epoch_samples

        # deep copy the model
        if phase == 'val':
          if epoch_loss - best_loss < -PARAMS['min_improvement']:
            no_improvement = 0
            print("Val loss improved by {}. Saving best model.".format(best_loss-epoch_loss))
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
          else:
            no_improvement += 1
            print("No loss improvement since {}/{} epochs.".format(no_improvement,patience))
            
      time_elapsed = time.time() - since
      print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
      if patience >= 0 and no_improvement > patience:
        break
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

  #model training
  optimizer_ft = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'])
  model = train_model(model, dataloaders, optimizer_ft, device, num_epochs=PARAMS['epochs'], patience=PARAMS['patience'])

  # save weights
  torch.save(model.state_dict(),"state_dict.pth")

if PARAMS["task"] in ["predict", "all"]:
  # load weights
  model.load_state_dict(torch.load("state_dict.pth", map_location="cpu"))
  device = torch.device('cpu')
  model = model.to(device)
  # denormalization function


  # visualize example segmentation
  import math
  model.eval()   # Set model to evaluate mode
  test_dataset = test_set#DenoiseDataset(test_dir, img_size=PARAMS['img_size'], count=PARAMS["test_dataset_size"])
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
  inputs, gts = next(iter(test_loader))
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
  fig = plot_side_by_side(x_ort, x_dem, y_dem_gt, y_dem_pr)

if PARAMS["neptune"]:
  neptune.log_image('input-gt-output', fig, image_name='input-gt-output')
  neptune.stop()

