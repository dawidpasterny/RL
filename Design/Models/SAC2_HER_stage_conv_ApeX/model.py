import numpy as np

import copy
import torch
import torch.nn as nn

HID_SIZE = 256
NUM_FEATURES = 21

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
        """ - screen is the pixel observation
            - obs is the concatenated current and target state

            Returns: combined features of the entire observation
        """
        c = self.conv(screens).view(screens.size()[0],-1)
        c = torch.column_stack([self.fc1(c), states])
        # return self.fc2(c)
        return c


class ActorSAC(nn.Module):
    def __init__(self, act_size, fe=None):
        super().__init__()
        self.fe = fe
        self.mu = nn.Sequential(
            nn.Linear(NUM_FEATURES, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus()
        )

        self.logstd = nn.Parameter(1e-3*torch.ones(act_size)) # for global variance
        # self.logstd = nn.Sequential(
        #     nn.Linear(NUM_FEATURES, HID_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(HID_SIZE, act_size)
        # )

    def forward(self, screen, state):
        x = self.fe(screen, state)         
        # return self.mu(x), self.std(x)
        return self.mu(x), torch.exp(self.logstd)
        # return self.mu(x), .05


class CriticSAC(nn.Module):
    """ Double-Q value estimator """
    def __init__(self, act_size, fe=None):
        super().__init__()
        self.fe = fe
        self.q1 = nn.Sequential(
            nn.Linear(NUM_FEATURES+act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(NUM_FEATURES+act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, screen, obs, act):
        x = torch.cat([self.fe(screen, obs), act], dim=1)
        return self.q1(x), self.q2(x)


class TargetNet():
    """ Just a wrapper with syncing functionality for target networks"""
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def __call__(self, *inpt):
        return self.target_model(*inpt)

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