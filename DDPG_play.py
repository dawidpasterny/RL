import sys
import os
sys.path.append(os.getcwd())

import gym
from lib import model
import stage_creator as sc

import numpy as np
import torch




if __name__ == "__main__":
    env = sc.StageCreator()
    obs_size = env.observation_space.shape[0]
    res = 64
    env = sc.ScreenOutput(res, env)
    device = torch.device("cpu")

    ae = model.Autoencoder(1, pretrained="./Autoencoder-FC.dat").float().to(device)
    obs_size += ae.get_bottleneck_size(res)[1]
    act_net = model.DDPGActor(obs_size, env.action_space.shape[0])
    act_net.load_state_dict(torch.load("./Actor-best.dat", map_location=torch.device(device)))
    
    screen, state = env.reset()
    env.render(ae=ae, delay=.5)
    total_reward = 0.0
    total_steps = 0
    episode = 1
    while episode<10:
        state_t = torch.tensor([state]).float()
        screen_t = torch.tensor([screen])
        features = torch.reshape(ae.encode(screen_t), (1,-1)).float()
        action_t = act_net(torch.column_stack((features, state_t))) # actions tensor
        action = action_t.squeeze(dim=0).data.numpy()
        print(action)
        # action = np.clip(action, -1, 1)
        (new_screen, new_state), reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        env.render(ae=ae, delay=.5)
        if done:
            episode += 1
            screen, state = env.reset()
        print(f"Reward: {total_reward}, steps: {total_steps}")