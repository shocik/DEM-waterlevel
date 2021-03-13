import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

vgg16_pretrained = models.vgg16(pretrained=True)

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(4, 32, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 64, 3, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            nn.BatchNorm2d(64),


        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(int(64*input_size/4*input_size/4),16),
            torch.nn.Linear(16,int(64*input_size/4*input_size/4))
        )
        self.decoder = torch.nn.Sequential(
            #Print(),
            torch.nn.ConvTranspose2d(64,64, 3, stride=2,padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            #Print(),
            torch.nn.ConvTranspose2d(64,32, 3, stride=2,padding=1, output_padding=1),
            torch.nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            #Print(),
            torch.nn.ConvTranspose2d(32, 1, 3, padding=1),
            #Print()
        )

        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        shape = tuple(x.size())
        x = torch.flatten(x,1,-1)
        #print(x.shape)
        x = self.dense(x)
        #print(x.shape)
        x = torch.reshape(x,shape)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        x = self.activation(x)
        return x  