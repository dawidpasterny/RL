import numpy as np

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 128

class Autoencoder(nn.Module):
    def __init__(self, io_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(io_size, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

class TargetNet():
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


