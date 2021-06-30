import sys
import os
sys.path.append(os.getcwd())
import numpy as np

from Design.Models.AE import model
import Design.Environments.stage_creator as sc

from stable_baselines import HER, SAC
from stable_baselines.common.env_checker import check_env

PATH = "./Design/Models/"
RES = 64

GAMMA = 0.99
BATCH_SIZE = 124
LEARNING_RATE = 1e-4
REPLAY_SIZE = 80000
REPLAY_INITIAL = 8000
TEST_INTERV = 1000
UNROLL = 2


device = "cpu"
ae = model.Autoencoder(1, pretrained=PATH+"AE/Autoencoder-FC.dat", device=device).to(device).float().eval()
env = sc.StageCreator()
env = sc.ScreenOutput(RES, env, ae)

check_env(env)


# model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
#             goal_selection_strategy='future',
#             verbose=1, buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.95, batch_size=256,
#             policy_kwargs=dict(layers=[256, 256, 256]))

# # Train for 1e5 steps
# model.learn(int(1e5))
# # Save the trained agent
# model.save(PATH+'SAC+HER_stage/SAC_HER-best')