import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#vgg16_pretrained = models.vgg16(pretrained=True)

class VggUnetLevel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(4, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            # save 1
            torch.nn.MaxPool2d(2, stride=2),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            # save 2
            torch.nn.MaxPool2d(2, stride=2),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            # save 3
            torch.nn.MaxPool2d(2, stride=2),
            # conv4
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            # save 4
            #torch.nn.MaxPool2d(2, stride=2),
            # conv5
            #torch.nn.Conv2d(512, 512, 3, padding=1),
            #torch.nn.ReLU(),
            #torch.nn.Conv2d(512, 512, 3, padding=1),
            #torch.nn.ReLU(),
            #torch.nn.Conv2d(512, 512, 3, padding=1),
            #torch.nn.ReLU(),
            ##torch.nn.MaxPool2d(2, stride=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(int(512*input_size/8*input_size/8),1)
            #torch.nn.Linear(int(input_size*input_size),2)
            #torch.nn.Linear(1024,1024),
            #torch.nn.Linear(1024,int(128*input_size/8*input_size/8))
        )
        # initialize weights
        #for i in range(len(self.encoder)):
        #    if isinstance(self.encoder[i], torch.nn.Conv2d):
        #        self.encoder[i].weight.data = vgg16_pretrained.features[i].weight.data
        #        self.encoder[i].bias.data = vgg16_pretrained.features[i].bias.data
        self.decoder = torch.nn.Sequential(
            # upconv4
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # concat 256
            # upconv3
            nn.Conv2d(512+256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # concat 128
            # upconv2
            nn.Conv2d(256+128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # concat 64
            # upconv1
            nn.Conv2d(128+64+1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # conv last
            nn.Conv2d(64, 1, 3, padding=1)
        )

        self.conv_out = dict()
        #self.sigmoid_activation = nn.Sigmoid()
    
    def forward(self, x):
        #block index initialization
        i=0
        self.conv_out[i] = torch.narrow(x, 1, 0, 1)
        #print(self.conv_out[i])
        #forward encoder
        for layer in self.encoder:
            if isinstance(layer, torch.nn.MaxPool2d):
                i+=1 # 1 -> 5
                self.conv_out[i] = x
                x = layer(x)
            else:
                x = layer(x)
        #forward decoder
        #for layer in self.decoder:
        #    if isinstance(layer, torch.nn.Upsample):
        #        x = layer(x)
        #        x = torch.cat([x, self.conv_out[i]], dim=1)
        #        if i==1:
        #          x = torch.cat([x, self.conv_out[0]], dim=1)
        #        i-=1 # 5 -> 1
        #    else:
        #        x = layer(x)

        x = torch.flatten(x,1,-1)
        x = self.dense(x)
        #x[:,0] = self.sigmoid_activation(x[:,0])
        #x = self.activation(x)
        return x