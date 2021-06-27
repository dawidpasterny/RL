import sys
import os
sys.path.append(os.getcwd())

import gym
from lib import model
import stage_creator as sc

import numpy as np
import torch
import argparse
from A3C import NUM_FEATURES, RES, MAX_STEPS
from model import FE, AE, AC

PATH = "./A3C/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-j", "--job", default=0)
    args = parser.parse_args()

    env = sc.StageCreator(seed=args.seed)
    obs_size = env.observation_space.shape[0]
    env = sc.ScreenOutput(RES, env)
    device = torch.device("cpu")

    # Models
    fe = FE((1,RES,RES), NUM_FEATURES).float()
    ae = AE(fe).float()
    alice = AC(NUM_FEATURES+6, 2, fe).float()
    bob = AC(NUM_FEATURES+6, 2, fe).float()
    fe.load_state_dict(torch.load(PATH+f"FE-{args.job}.dat", map_location=torch.device(device)))
    ae.load_state_dict(torch.load(PATH+f"AE-{args.job}.dat", map_location=torch.device(device)))
    alice.load_state_dict(torch.load(PATH+f"Alice-{args.job}.dat", map_location=torch.device(device)))
    bob.load_state_dict(torch.load(PATH+f"Bob-{args.job}.dat", map_location=torch.device(device)))

    # Bob plays with randomly generated target
    # screen, state = env.reset(target=True)
    # env.render(ae=ae, delay=.5)
    # steps = 0.0
    # episode = 1

    # while episode<10:
    #     print("State: ", state)
    #     state_t = torch.FloatTensor([state])
    #     screen_t = torch.FloatTensor([screen])
    #     mu, _, _, _ = bob(screen_t, state_t) # just take the distributions mean, no exploration now
    #     action = mu.data.numpy()
    #     print("Action: ", *action)
         
    #     (new_screen, new_state), reward, done, _ = env.step(*action)
    #     print("Next state: ", new_state)
    #     steps += 1
    #     env.render(ae=ae, delay=.5)
    #     if done:
    #         print(f"Episode: {episode} | reward: {reward}, steps: {steps}")
    #         steps = 0.0
    #         episode += 1
    #         screen, state = env.reset(target=True)

    # Alice generates a target
 
    steps, a_steps = 0, 0
    episode = 1

    while episode<5:
        screen, state = env.reset()
        env.render(ae=ae, delay=1)
        print("Alice's turn")
        while True:
            a_steps += 1
            state_t = torch.FloatTensor([state])
            screen_t = torch.FloatTensor([screen])

            # Samples action
            mu, var, _, p = alice(screen_t, state_t)
            stop = bool(torch.bernoulli(p))
            action = mu.data.numpy()
            print("Action: ", list(action[0]))

            # Perform step +[1] to indicate that it's alice
            (screen, state), _, done, _ = env.step(list(action[0])+[1])
            env.render(ae=ae, delay=1)
            if (stop and len(env.traj)>1) or a_steps>=MAX_STEPS:
                # Stopping at a right moment is what Alice needs to learn actually
                if stop:
                    print(f"Alice stopped after {len(env.traj)} gears, {a_steps} steps")
                break
                
        target = state
        env.set_target(target[:2], target[2])
        screen, state = env.reset(random=False)
        env.render(ae=ae, delay=1)

        print("Bob's turn")
        print("State: ", state)
        state_t = torch.FloatTensor([state])
        screen_t = torch.FloatTensor([screen])
        mu, _, _, _ = bob(screen_t, state_t) # just take the distributions mean, no exploration now
        action = mu.data.numpy()
        print("Action: ", *action)
         
        (new_screen, new_state), reward, done, _ = env.step(*action)
        print("Next state: ", new_state)
        steps += 1
        env.render(ae=ae, delay=1)
        if done:
            print(f"Episode: {episode} | reward: {reward}, steps: {steps}")
            steps, a_steps = 0, 0
            episode += 1

    
