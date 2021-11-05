import numpy as np

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, io_size, pretrained=None, device="cpu"):
        super().__init__()
        self.device = device
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
        # sigmoid to normalize the features to the same order of magniture as state
        # x = torch.sigmoid(x.view(x.shape[0],-1))
        # x = torch.tanh(x.view(x.shape[0],-1))
        # print(x)
        return x

    def decode(self,x):
        # x = nn.functional.relu(self.de_fc(x))
        # x = torch.reshape(x, (1,32,9,9))
        x = self.decoder(x)
        return x

    def get_bottleneck_size(self, shape):
        o = self.encoder(torch.zeros(1, shape, shape).to(self.device).unsqueeze(dim=1))
        o = torch.reshape(o, (1,-1))
        return o.size()


class Autoencoder84(nn.Module):
    def __init__(self, io_size, pretrained=None, device="cpu"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ConstantPad2d(1, 1.0),
            # PrintLayer(),
            nn.Conv2d(io_size, 16, kernel_size=8, stride=3),
            nn.ReLU(),
            # PrintLayer(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), 
            nn.ReLU(),
            # PrintLayer(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # PrintLayer(),
            nn.Conv2d(64, 84, kernel_size=5),
            # PrintLayer(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(84, 64, 5),
            nn.ReLU(),
            # PrintLayer(),
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.ReLU(),
            # PrintLayer(),
            nn.ConvTranspose2d(32, 16, 5, 2),
            nn.ReLU(),
            # PrintLayer(),
            nn.ConvTranspose2d(16, 1, 8, 3, padding=1),
            # PrintLayer(),
            nn.Sigmoid()
        )

        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __call__(self, x): # feature extractor
        x = self.encoder(x)
        x = x.view(x.shape[0],-1)
        # x = self.en_fc(x)
        return torch.sigmoid(x) # to normalize the features to the same order of magniture as state

    def get_bottleneck_size(self, shape):
        o = self.encoder(torch.zeros(1, shape, shape).unsqueeze(dim=1))
        o = torch.reshape(o, (1,-1))
        return o.size()


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x



if __name__=="__main__":
    x = torch.randn(1,1,84,84)
    ae = Autoencoder84(1)
    output = ae.forward(x)