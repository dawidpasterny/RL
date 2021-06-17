import numpy as np

import torch
import torch.nn as nn



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


