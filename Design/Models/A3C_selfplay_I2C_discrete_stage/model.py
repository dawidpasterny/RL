import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

NUM_FEATURES = 24
HID_SIZE = 256


class FE(nn.Module):
    """ Convolutional feature extractor to be shared among various networks """
    def __init__(self, screen_shape, state_size):
        super().__init__()
        self.screen_shape = screen_shape
        self.conv = nn.Sequential(
            nn.Conv2d(screen_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out()
        self.fc1 = nn.Sequential(
            nn.Linear(conv_out_size, NUM_FEATURES - state_size),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(NUM_FEATURES, NUM_FEATURES),
            nn.Softplus()
        )

    def _get_conv_out(self):
        o = self.conv(torch.zeros(1, *self.screen_shape))
        return int(np.prod(o.size()))

    def forward(self, screens, states):
        c = self.conv(screens).view(screens.size()[0],-1)
        c = torch.column_stack([self.fc1(c), states])
        # return self.fc2(c)
        return c


class AC(nn.Module):
    def __init__(self, d_act_size, phi_act_size, fe):
        super().__init__()
        self.fe=fe

        self.actor = nn.Sequential( 
            nn.Linear(NUM_FEATURES, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, phi_act_size*d_act_size) # joint prob. distr ovef d and phi
        )

        self.critic = nn.Sequential(
            nn.Linear(NUM_FEATURES, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

        self.p = nn.Sequential(
            nn.Linear(NUM_FEATURES, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 2), # binary output
        )


    def _get_conv_out(self, screen_shape):
        o = self.conv(torch.zeros(1, *screen_shape))
        return int(np.prod(o.size()))


    def forward(self, screens, states):
        c = self.fe(screens, states)
        logits = self.actor(c)
        val = self.critic(c)
        stop = self.p(c)
        return logits, val, stop


