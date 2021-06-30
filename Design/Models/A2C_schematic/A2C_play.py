import sys
import os
sys.path.append(os.getcwd())

import time
import argparse
import numpy as np
import gym

import torch
import collections

from Design.Environments import y_schematic_reward as ys
from Design.Models.A2C.A2C import PolicyAgent, A2CNet

# python ./Design/Models/A2C/A2C_play.py -m ./Design/Models/A2C/runs/A2C2D-best-A2C2D-best-A2C2DMar25_13_21_08.dat
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-s", "--seed", default=0)
    parser.add_argument("-v","--visualize", default=True)
    args = parser.parse_args()

    env = ys.GridSchematic3D()
    net = A2CNet(env.observation_space.shape[0], sum([a.n for a in env.action_space]))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    agent = PolicyAgent(lambda x: net(x)[0])
    episodes=10

    print(f"Current y= {env.y}, distance to target= {env.y_target-env.y}")
    i_dist = abs(1.0 - env.i_target)
    print(f"Current ratio= {env.i}, distance to target= {i_dist}")

    state = env.reset()
    while episodes>0:
        action = agent([state])
        new_state, reward, done = env.step(action[0])
        y_dist, i_dist, _ = new_state
        state = new_state

        print(f"Current y {env.y}, distance to target {y_dist}")
        print(f"Current ratio {env.i}, distance to target {i_dist}")
        
        if args.visualize:
            env.render()
        if done:
            env.render(path=f"./pics/game{episodes}_")
            if env.static:
                break
            else:
                state = env.reset()
                episodes -= 1
        
