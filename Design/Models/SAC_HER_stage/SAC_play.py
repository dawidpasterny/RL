import sys
import os
import glob
sys.path.append(os.getcwd())

import numpy as np
import torch

from Design.Models.AE.model import Autoencoder84
from Design.Models.SAC_HER_stage.model import ModelActor
import Design.Environments.stage_creator_unwrapped as sc


PATH = "./Design/Models/"
RES = 84

device = "cuda" if torch.cuda.is_available() else "cpu"
ae = Autoencoder84(1, pretrained=PATH+"AE/Autoencoder84.dat", device=device).to(device).float().eval()
env = sc.StageCreator(RES, ae=ae)
obs_size = env.observation_space["observation"].shape[0] + \
            env.observation_space["desired_goal"].shape[0]
act_size = env.action_space.shape[0]

act_net = ModelActor(obs_size, act_size).to(device)
act_net.load_state_dict(torch.load(PATH+f"SAC_HER_stage/Actor_best-1.dat", map_location=torch.device(device)))

obs = env.reset()["observation"]
target = env.goal
obs = np.hstack((obs, target))
env.render()

steps = 0
episode = 1

files = glob.glob(PATH + f"SAC_HER_stage/test_pic/*.png")
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

print("std:", torch.exp(act_net.logstd))

while episode<10:
    obs_t = torch.FloatTensor([obs])
    mu, _ = act_net(obs_t)
    action = mu.detach().numpy()[0] # mean action
    print("Action:", action)
    
    # Perform step
    next_obs, reward, done, _ = env.step(action)
    obs = np.hstack((next_obs["observation"], target))
    env.render(path=PATH + f"SAC_HER_stage/test_pic/EP{episode}IT{steps}")
    steps += 1

    if done:
      print(f"Episode: {episode} | reward: {reward}, steps: {steps}")
      obs = env.reset()["observation"]
      target = env.goal
      obs = np.hstack((obs, target))
      episode += 1
      steps = 0
