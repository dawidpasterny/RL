import numpy as np

import copy
import torch
import torch.nn as nn

HID_SIZE = 256
x_0 = 200
k = 1e-2
logist = lambda x: .05 + 0.7/(1+torch.exp(-k*(x-x_0))) # arbitrary logistic curve


class Logist(nn.Module):
    """ Arbitrary logistic curve b + a/(1+exp(-k(x-x_0)))
    """
    def __init__(self, a, b, k, x_0):
        super().__init__() # init the base class
        self.a = a
        self.b = b
        self.k = k
        self.x_0 = x_0

    def forward(self, x):
        # return self.b + self.a*torch.sigmoid(self.k*(x-self.x_0))
        return self.b + self.a/(1+torch.exp(-self.k*(x-self.x_0)))


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            # nn.BatchNorm1d(HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            # nn.BatchNorm1d(HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
            # nn.Softplus()
            # Logist(.05, .7, 1e-2, 200)
            # .05 + 0.7*nn.Sigmoid()
        )

        self.logstd = nn.Parameter(-3*torch.ones(act_size)) # for global variance

        # self.logstd = nn.Sequential(
        #     nn.Linear(obs_size, HID_SIZE),
        #     nn.BatchNorm1d(HID_SIZE),
        #     nn.Tanh(),
        #     nn.Linear(HID_SIZE, HID_SIZE),
        #     nn.BatchNorm1d(HID_SIZE),
        #     nn.Tanh(),
        #     nn.Linear(HID_SIZE, act_size)
        # )

    def forward(self, x):
        # return self.mu(x), torch.exp(self.logstd(x)) + 1e-6 # for stability
        # return logist(self.mu(x)), torch.exp(self.logstd)
        # return self.mu(x), self.logstd
        return logist(self.mu(x)), .08

    
# class ModelActor(nn.Module):
#     def __init__(self, obs_size, act_size):
#         super(ModelActor, self).__init__()

#         self.d = nn.Sequential(
#             nn.Linear(obs_size, HID_SIZE),
#             nn.ReLU(),
#             nn.Linear(HID_SIZE, HID_SIZE),
#             nn.ReLU(),
#             nn.Linear(HID_SIZE, act_size),
#             nn.Softmax()
#         )

#         self.phi = nn.Sequential(
#             nn.Linear(obs_size, HID_SIZE),
#             nn.ReLU(),
#             nn.Linear(HID_SIZE, HID_SIZE),
#             nn.ReLU(),
#             nn.Linear(HID_SIZE, act_size),
#             nn.Softmax()
#         )

#     def forward(self, x):
#         return self.d(x), self.phi(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


class ModelSACTwinQ(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelSACTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)


class TargetNet():
    """ Just a wrapper with syncing functionality for target networks"""
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def __call__(self, inpt):
        return self.target_model(inpt)

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