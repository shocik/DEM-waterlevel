# -*- coding: utf-8 -*-

import os
#set workdir
#os.chdir("/content/drive/MyDrive/RiverSemanticSegmentation/")

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

#training parameters in neptune format
PARAMS = {
    "img_size": 416,
    "model": "vgg_unet",
    "learning_rate": 0.0001,
    "batch_size": 8,
    'epochs': 1000,
    'patience': 10,
    "train_dataset_size": -1, # set train dataset subset. Useful when neet to 
                              # overtrain model with small amount of images.
                              # -1 -all images from train directories.
    "test_dataset_size": -1,  # set test dataset subset.
                              # -1 -all images from train directories.
    'image_preload': False,
}

#neptune installation and initialization
#!pip install neptune-client
import neptune
neptune.init(project_qualified_name='radek/denoise1',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYmY4YjQ3YjEtNmY5My00MDc2LWI4NzAtMWE5MmUwZjQ1NDE2In0=',
             )
neptune.create_experiment(params=PARAMS)

#dataset configuration
dataset_dir = os.path.normpath("dataset")
train_dir = os.path.join(dataset_dir,"train")
test_dir = os.path.join(dataset_dir,"test")

train_set = DenoiseDataset(train_dir, img_size=PARAMS['img_size'], count=PARAMS["train_dataset_size"])
test_set = DenoiseDataset(test_dir, img_size=PARAMS['img_size'], count=PARAMS["test_dataset_size"])

batch_size = PARAMS['batch_size']
dataloaders = {
    'train': DataLoader(train_set, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0),
    'val': DataLoader(test_set, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0)
}

# load images - useful if you want to save some time by preloading images (very time-consuming) when 
# the model is still not fuctional and cant run standard training.
if PARAMS['image_preload']:
  for phase in dataloaders:
    for inputs, labels in tqdm(dataloaders[phase]):
      pass

#model loading
elif PARAMS['model'] == "vgg_unet":
  from models.vgg_unet import VggUnet
  model = VggUnet()


#model structure preview
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model_stats = summary(model, input_size=(PARAMS['batch_size'], 4, PARAMS['img_size'], PARAMS['img_size']))
for line in str(model_stats).splitlines():
  neptune.log_text('model_summary', line)

from collections import defaultdict
import torch.nn.functional as F
SMOOTH = 1e-6
def iou_metric(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs[:,1,:,:]  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels[:,1,:,:]
    intersection = (outputs * labels).sum(2).sum(1)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs + labels).sum(2).sum(1) - intersection  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou.mean()

def calc_loss(pred, target, metrics):
    loss = F.mse_loss(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

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
    best_loss = 0
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
                model.eval()   # Set model to evaluate mode

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
              if epoch_loss < best_loss:
                no_improvement = 0
                print("Val IoU improved by {}. Saving best model.".format(best_loss-epoch_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
              else:
                no_improvement += 1
                print("No accuracy improvement since {}/{} epochs.".format(no_improvement,patience))
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if patience >= 0 and no_improvement > patience:
          break
    print('Best accuracy: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#model training
optimizer_ft = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'])
model = train_model(model, dataloaders, optimizer_ft, device, num_epochs=PARAMS['epochs'], patience=PARAMS['patience'])

# save weights
torch.save(model.state_dict(),"state_dict.pth")

neptune.log_artifact('state_dict.pth')

# load weights
model.load_state_dict(torch.load("state_dict.pth", map_location="cpu"))

# denormalization function
from torchvision import transforms
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

def reverse_transform(inp):
    print(inp.shape)
    inp = inv_normalize(inp)
    inp = inp.numpy()
    inp = np.swapaxes(inp, 1, 3)
    inp = np.swapaxes(inp, 1, 2)
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp
def labels2mask(labels):
    return labels[:,1,:,:]

# helper function to plot input, ground truth and predict images in grid
import matplotlib.pyplot as plt
def plot_side_by_side(input,y_dem_gt, y_dem_pr):
  assert input.shape[0] == y_dem_gt.shape[0] == y_dem_pr.shape[0]
  batch_size = input.shape[0]
  fig, axs = plt.subplots(batch_size, 4, figsize=(30,50))
  for i in range(batch_size):
    axs[i, 0].imshow(np.moveaxis(input[i,1:4], 0, -1))
    axs[i, 1].imshow(input[i,0])
    axs[i, 2].imshow(y_dem_gt)
    axs[i, 3].imshow(y_dem_pr)

# visualize example segmentation
import math
model.eval()   # Set model to evaluate mode
test_dataset = DenoiseDataset(test_dir, img_size=PARAMS['img_size'], count=PARAMS["test_dataset_size"])
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
inputs, gts = next(iter(test_loader))
inputs = inputs.to(device)
gts = gts.to(device)

gts = gts.data.cpu().numpy()
pred = model(inputs)

pred = pred.data.cpu().numpy()
inputs = inputs.data.cpu()

# use helper function to plot
plot_side_by_side(inputs, gts, pred)

#evaluate model
#test_dataset = DenoiseDataset(x_test_dir, y_test_dir, input_size=PARAMS['input_size'], output_size=PARAMS['output_size'], n_classes=PARAMS['n_classes'])
#test_loader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"], shuffle=True, num_workers=0)
#intersection=0
#union=0
#for inputs, labels in tqdm(test_loader):
#  inputs = inputs.to(device)
#  labels = labels.to(device)
#  labels = labels.data.cpu().numpy()
#  pred = model(inputs)
#  pred = torch.round(pred)
#  pred = pred.data.cpu().numpy()
#  target = labels[:,1,:,:]
#  predict = pred[:,1,:,:]
#  temp = (target * predict).sum()
#  intersection+=temp
#  union+=((target + predict).sum() - temp)
#iou = intersection/union
#print("IoU: {}".format(iou))
#neptune.log_metric("total_iou",iou)

# update neptune status
neptune.stop()