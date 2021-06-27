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


class FE(nn.Module):
    """ Convolutional feature extractor to be shared among various networks """
    def __init__(self, input_shape, n_features):
        super().__init__()
        self.input_shape = input_shape
        self.n_features = n_features
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # 1 channel data
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out()
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, n_features),
            nn.Sigmoid()
        )

    def _get_conv_out(self):
        o = self.conv(torch.zeros(1, *self.input_shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        c = self.conv(x).view(x.size()[0],-1)
        return self.fc(c)
    
    
class AE(nn.Module):
    def __init__(self, fe):
        super().__init__()
        self.fe = fe
        self.conv_in_shape = fe.conv(torch.zeros(1,*fe.input_shape)).size()
        print(self.conv_in_shape)
        
        self.decoder = nn.Sequential(
            nn.Linear(fe.n_features, np.prod(self.conv_in_shape)),
            nn.ReLU(),
            nn.Unflatten(1,self.conv_in_shape[1:]),           
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, fe.input_shape[0], kernel_size=8, stride=4), # 1 channel data
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.fe(x)
        return self.decoder(x)
            


class AC(nn.Module):
    def __init__(self, obs_size, act_size, fe):
        super().__init__()
        self.fe=fe

        self.fc = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU()
        )

        self.mu = nn.Sequential( 
            nn.Linear(256, act_size),
            nn.Sigmoid() 
        )

        self.var = nn.Sequential(
            nn.Linear(256, act_size),
            nn.Softplus(),
        )

        self.val = nn.Linear(256, 1)

        self.p = nn.Sequential( # p of Bernouli distribution
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def forward(self, screens, states):
        c = self.fe(screens) # extract features from pixel input
        c = self.fc(torch.column_stack([c, states]))
        mu = self.mu(c)
        var = self.var(c)
        v = self.val(c)
        p = self.p(c)
        return mu, var, v, p


