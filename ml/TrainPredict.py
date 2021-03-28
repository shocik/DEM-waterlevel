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

class TrainPredict:
    def __init__(self, PARAMS):
        print(PARAMS)
        self.PARAMS = PARAMS
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

        train_set = DenoiseDataset(train_dir, img_size=PARAMS['img_size'], augment="True", repeat=20)
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
