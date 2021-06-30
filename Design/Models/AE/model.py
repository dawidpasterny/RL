import numpy as np

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, io_size, pretrained=None, device="cpu"):
        super().__init__()
        self.encoder = nn.Sequential( # 1x128x128 -> 32x10x10
            # nn.MaxPool2d(2,2),
            nn.Conv2d(io_size, 8, kernel_size=4, stride=2), # 31
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2), # 15
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), # 7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7) # 1
        )

        self.decoder = nn.Sequential( # 32x10x10 -> 1x128x128
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2),
            nn.Sigmoid()
        )

        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))

    def forward(self,x):
        x = self.encoder(x)
        x = self.decode(x)
        return x

    def __call__(self, x): # feature extractor
        x = self.encoder(x)
        x = x.view(x.shape[0],-1)
        # x = self.en_fc(x)
        return torch.sigmoid(x) # to normalize the features to the same order of magniture as state

    def decode(self,x):
        # x = nn.functional.relu(self.de_fc(x))
        # x = torch.reshape(x, (1,32,9,9))
        x = self.decoder(x)
        return x

    def get_bottleneck_size(self, shape):
        o = self.encoder(torch.zeros(1, shape, shape).unsqueeze(dim=1))
        o = torch.reshape(o, (1,-1))
        return o.size()


class Autoencoder84(nn.Module):
    def __init__(self, io_size, pretrained=None, device="cpu"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ConstantPad2d(1, 1),
            nn.Conv2d(io_size, 8, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 7, 3),
            nn.Sigmoid()
        )

        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))

    def forward(self,x):
        x = self.encoder(x)
        x = self.decode(x)
        return x

    def __call__(self, x): # feature extractor
        x = self.encoder(x)
        x = x.view(x.shape[0],-1)
        # x = self.en_fc(x)
        return torch.sigmoid(x) # to normalize the features to the same order of magniture as state

    def decode(self,x):
        # x = nn.functional.relu(self.de_fc(x))
        # x = torch.reshape(x, (1,32,9,9))
        x = self.decoder(x)
        return x

    def get_bottleneck_size(self, shape):
        o = self.encoder(torch.zeros(1, shape, shape).unsqueeze(dim=1))
        o = torch.reshape(o, (1,-1))
        return o.size()

           