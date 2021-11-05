import sys
import os
import glob
sys.path.append(os.getcwd())

import numpy as np
import torch

from Design.Models.AE.model import Autoencoder84
import Design.Models.SAC2_HER_stage_conv_ApeX.model as model
import Design.Environments.discrete_stage_creator_unwrapped as sc
from Design.Models.SAC2_HER_stage_conv_ApeX.SAC import PATH, RES


env = sc.StageCreator(RES)
screen_shape = obs_size = env.observation_space["observation"]["screen"].shape
state_size = 7 # x, y, i, d, x_target, y_target, i_target
act_size = 2
device = "cpu"

fe = model.FE(screen_shape, state_size)
act_net = model.ActorSAC(act_size, fe).to(device)
# crt_net = model.CriticSAC(act_size, fe).to(device)
# tgt_crt_net = model.TargetNet(crt_net)

act_net.load_state_dict(torch.load(PATH+f"Actor_last-2.dat", map_location=torch.device(device)))
# crt_net.load_state_dict(torch.load(PATH+f"Critic_last-0.dat", map_location=torch.device(device)))

# Cleanup the pic folder from previous run
files = glob.glob(PATH + f"test_pic/*.png")
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

screen, obs = env.reset(random=True)["observation"] # we want to test generalization so random=True
env._set_reachable_target()
obs = np.hstack((obs, env.goal))
env.render()
print("std:", torch.exp(act_net.logstd))

steps = 0
episode = 1

while episode<10:
    print("Init state: ", obs)
    state_t = torch.FloatTensor(np.array([obs])).to(device)
    screen_t = torch.FloatTensor(np.array([screen])).to(device)
    mu, _ = act_net(screen_t, state_t) # just take the expected action
    mu = mu.detach().numpy()[0]
    action = [sc.DIAMS[min(max(2,round(mu[0]/0.05)),5)-2], sc.ANGLES[min(round(mu[1]/(np.pi/3)),5)]] 
    print(f"d={mu[0]:.2f}, phi={mu[1]:.2f}")
    print(f"action[0]={action[0]}, action[1]={action[1]}")

    # Perform step
    new_obs, reward, done, _ = env.step(action)
    obs = np.hstack((new_obs["observation"][1], env.goal))
    screen = new_obs["observation"][0]

    env.render(path=PATH + f"test_pic/EP{episode}IT{steps}")
    print("Next state: ", obs)

    steps += 1
    if done:
        print(f"Episode: {episode} | reward: {reward}, steps: {steps}")
        screen, obs = env.reset(random=True)["observation"] # we want to test generalization so random=True
        env._set_reachable_target()
        obs = np.hstack((obs, env.goal))
        env.render()
        episode += 1
        steps = 0
