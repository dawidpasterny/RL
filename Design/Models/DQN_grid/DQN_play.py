import sys
import os
sys.path.append(os.getcwd())

import time
import argparse
import numpy as np
import gym

import torch
import collections

from Design.Environments import grid_schematic_2D as g2d
from Design.Models.DQN import DQN

DEFAULT_ENV_NAME = "24x24Wall"
FPS = 25

# python3 ./Design/Models/DQN_play.py -m ./Design/Models/Grid2dCNN-best.dat -s 3672871121734420758 -r ./Design/Models
# tensorboard --logdir=Design/Models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-s", "--seed", default=3672871121734420758)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    # if isinstance(args.env, str):
    #     env = g2d.GridSchematic2D(map_name = args.env)
    # else:
    # env = g2d.GridSchematic2D(seed = int(args.seed))
    env = g2d.GridSchematic2D(seed = 3672871121734420758)
    env = g2d.ScreenOutput(env)

    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)
    net = DQN(env.observation_space.shape, env.action_space.n).float()
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()