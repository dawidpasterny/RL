import numpy as np

import copy
import torch
import torch.nn as nn


class FE(nn.Module):
    """ Convolutional feature extractor to be shared among various networks """
    def __init__(self, input_shape, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # 1 channel data
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, n_features),
            nn.Sigmoid()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        c = self.conv(x).view(x.size()[0],-1)
        return self.fc(c)


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, fe):
        super(DDPGActor, self).__init__()
        self.fe=fe
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Sigmoid()
        )

    def forward(self, screen, state):
        x = self.fe(screen)
        x = torch.column_stack([x, state])
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size, fe):
        super(DDPGCritic, self).__init__()
        self.fe=fe
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, screen, state, a):
        x = self.fe(screen)
        x = torch.column_stack([x, state])
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class TargetNet():
    """ Just a wrapper with syncing functionality for target networks"""
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def __call__(self, *args):
        self.target_model(args)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """ Soft sync, performs convex combination of target net parameters with
            model parameters
        """
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)

