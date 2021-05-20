import numpy as np

import copy
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
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x): # feature extractor
        x = self.encoder(x)
        # x = torch.flatten(x,1)
        # x = self.en_fc(x)
        return x

    def decode(self,x):
        # x = nn.functional.relu(self.de_fc(x))
        # x = torch.reshape(x, (1,32,9,9))
        x = self.decoder(x)
        return x

    def get_bottleneck_size(self, shape):
        o = self.encoder(torch.zeros(1, shape, shape).unsqueeze(dim=1))
        o = torch.reshape(o, (1,-1))
        return o.size()


# class Autoencoder(nn.Module):
#     def __init__(self, io_size, pretrained=False):
#         super().__init__()
#         self.encoder = nn.Sequential( # 1x128x128 -> 32x1x1
#             nn.Conv2d(io_size, 16, kernel_size=8, stride=3), #40
#             nn.ReLU(),
#             nn.Conv2d(16, 16, kernel_size=4, stride=2), #19
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2), # 9
#             nn.ReLU(),
#         )

#         self.en_fc = nn.Linear(9*9*32, 128)
#         self.de_fc = nn.Linear(128, 9*9*32)

#         self.decoder = nn.Sequential( # 32x6x6 -> 1x128x128
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2), # 19
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 16, 4, stride=2, output_padding=1), # 40
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, io_size, 8, 3),
#             nn.Sigmoid()
#         )

#         if pretrained:
#             self.load_state_dict(torch.load(pretrained))

#     def forward(self,x):
#         x = self.encode(x)
#         x = self.decode(x)
#         return x

#     def encode(self, x): # feature extractor
#         x = self.encoder(x)
#         x = torch.flatten(x,1)
#         x = self.en_fc(x)
#         return x

#     def decode(self,x):
#         x = nn.functional.relu(self.de_fc(x))
#         x = torch.reshape(x, (1,32,9,9))
#         x = self.decoder(x)
#         return x


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Sigmoid()
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


