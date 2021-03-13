# -*- coding: utf-8 -*-

import os
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
from helper import plot_side_by_side

#training parameters in neptune format
PARAMS = {
    "img_size": 256,
    "model": "vgg_unet",
    "learning_rate": 0.01,
    "batch_size": 2,
    'epochs': 1000,
    'patience': 10,
    'image_preload': False,
    'names': ['17.npy','12.npy']
}


#dataset configuration
dataset_dir = os.path.normpath("dataset")
train_dir = os.path.join(dataset_dir,"train")
test_dir = os.path.join(dataset_dir,"test")

train_set = DenoiseDataset(train_dir, img_size=PARAMS['img_size'], names=PARAMS['names'])

batch_size = PARAMS['batch_size']
dataloader = DataLoader(train_set, batch_size=PARAMS['batch_size'], shuffle=False, num_workers=0)

# load images - useful if you want to save some time by preloading images (very time-consuming) when 
# the model is still not fuctional and cant run standard training.
if PARAMS['image_preload']:
  for phase in dataloader:
    for inputs, labels in tqdm(dataloader[phase]):
      pass

#model loading
if PARAMS['model'] == "vgg_unet":
  from models.vgg_unet import VggUnet
  model = VggUnet()
elif PARAMS['model'] == "autoencoder":
  from models.autoencoder import Autoencoder
  model = Autoencoder(PARAMS['img_size'])

#model structure preview
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model_stats = summary(model, input_size=(PARAMS['batch_size'], 4, PARAMS['img_size'], PARAMS['img_size']))

from collections import defaultdict
import torch.nn.functional as F

def calc_loss(pred, target, metrics):
    loss = F.mse_loss(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples):   

    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        #neptune.log_metric(phase+"_"+k, metrics[k] / epoch_samples) #log
    print("{}: {}".format("train", ", ".join(outputs)))

#training loop
def train_model(model, dataloader, optimizer, device, num_epochs=25, patience=-1):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improvement = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase


        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
            
        model.train()  # Set model to training mode


        metrics = defaultdict(float)
        epoch_samples = 0
            
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)             

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, metrics)
                #print(model.encoder[0].weight.grad)
                # backward + optimize only if in training phase
 
                loss.backward()
                #pdb.set_trace()
                optimizer.step()

            # statistics
            epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model

            if epoch_loss < best_loss:
              no_improvement = 0
              print("Loss improved by {}. Saving best model.".format(best_loss-epoch_loss))
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
model = train_model(model, dataloader, optimizer_ft, device, num_epochs=PARAMS['epochs'], patience=PARAMS['patience'])

# save weights
torch.save(model.state_dict(),"state_dict.pth")


# load weights
model.load_state_dict(torch.load("state_dict.pth", map_location="cpu"))
device = torch.device('cpu')
model = model.to(device)
# denormalization function
from torchvision import transforms
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

def reverse_transform(inp):

    inp = inv_normalize(inp)
    inp = inp.numpy()
    inp = np.swapaxes(inp, 1, 3)
    inp = np.swapaxes(inp, 1, 2)
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp




# visualize example segmentation
import math
model.eval()   # Set model to evaluate mode
test_dataset = train_set#DenoiseDataset(test_dir, img_size=PARAMS['img_size'], count=PARAMS["test_dataset_size"])
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
inputs, gts = next(iter(test_loader))
inputs = inputs.to(device)
gts = gts.to(device)

gts = gts.data.cpu().numpy()
pred = model(inputs)

pred = pred.data.cpu().numpy()
inputs = inputs.data.cpu()

# use helper function to plot
plot_side_by_side(inputs, gts, pred)




