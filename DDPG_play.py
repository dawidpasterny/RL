import sys
import os
sys.path.append(os.getcwd())

import gym
from lib import model
import stage_creator as sc

import numpy as np
import torch
import argparse


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

    ae = model.Autoencoder(1, pretrained="./Autoencoder-FC.dat").float().to(device)
    obs_size += ae.get_bottleneck_size(res)[1]
    act_net = model.DDPGActor(obs_size, env.action_space.shape[0])
    act_net.load_state_dict(torch.load(f"Actor-best-{args.job}.dat", map_location=torch.device(device)))
    crt_net = model.DDPGCritic(obs_size, env.action_space.shape[0]).to(device).float()
    crt_net.load_state_dict(torch.load(f"Critic_best-{args.job}.dat", map_location=torch.device(device)))

    screen, state = env.reset()
    env.render(ae=ae, delay=.5)
    steps = 0.0
    episode = 1

    while episode<10:
        # print("State: ", state)
        state_t = torch.tensor([state]).float()
        screen_t = torch.tensor([screen])
        features = torch.reshape(ae(screen_t), (1,-1)).float()
        action_t = act_net(torch.column_stack((features, state_t))) # actions tensor
        action = action_t.squeeze(dim=0).data.numpy()
        print("Action: ", action)
        # stacked_state = torch.column_stack((torch.reshape(features.detach(), (1,-1)),state_t))
        # q_val = crt_net(stacked_state, action_t)
        # print("Q(s,a): ", q_val)
        
        (new_screen, new_state), reward, done, _ = env.step(action)
        # print("Next state: ", new_state)
        steps += 1
        env.render(ae=ae, delay=.1)
        if done:
            print(f"Episode: {episode} | reward: {reward}, steps: {steps}")
            steps = 0.0
            episode += 1
            screen, state = env.reset()
        
