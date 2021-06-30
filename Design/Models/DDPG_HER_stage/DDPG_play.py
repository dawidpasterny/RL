import sys
import os
sys.path.append(os.getcwd())

import gym
from Design.Models.DDPG_stage.lib import model
from Design.Models.AE import model as ae
import Design.Environments.stage_creator as sc

import numpy as np
import torch
import argparse

PATH = "./Design/Models/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-j", "--job", default=None)
    args = parser.parse_args()

    env = sc.StageCreator(seed=args.seed)
    obs_size = env.observation_space.shape[0]
    res = 64
    env = sc.ScreenOutput(res, env)
    device = torch.device("cpu")

    ae = ae.Autoencoder(1, pretrained=PATH+"AE/Autoencoder-FC.dat").float().eval().to(device)
    obs_size += ae.get_bottleneck_size(res)[1]
    act_net = model.DDPGActor(obs_size, env.action_space.shape[0], ae)
    act_net.load_state_dict(torch.load(PATH+f"DDPG_HER_stage/Actor-best-{args.job}.dat", map_location=torch.device(device)))
    crt_net = model.DDPGCritic(obs_size, env.action_space.shape[0], ae).to(device).float()
    crt_net.load_state_dict(torch.load(PATH+f"DDPG_HER_stage/Critic_best-{args.job}.dat", map_location=torch.device(device)))

    screen, state = env.reset()
    env.render(ae=ae, delay=.5)
    steps = 0.0
    episode = 1

    while episode<10:
        # print("State: ", state)
        state_t = torch.FloatTensor([state]).to(device)
        screen_t = torch.FloatTensor([screen]).to(device)
        action_t = act_net(screen_t, state_t) # actions tensor
        action = action_t.squeeze(dim=0).data.numpy()
        print("Action: ", action)
        # stacked_state = torch.column_stack((torch.reshape(features.detach(), (1,-1)),state_t))
        # q_val = crt_net(stacked_state, action_t)
        # print("Q(s,a): ", q_val)
        
        (new_screen, new_state), reward, done, _ = env.step(action)
        # print("Next state: ", new_state)
        steps += 1
        env.render(ae=ae, delay=1)
        if done:
            print(f"Episode: {episode} | reward: {reward}, steps: {steps}")
            steps = 0.0
            episode += 1
            screen, state = env.reset()
        
