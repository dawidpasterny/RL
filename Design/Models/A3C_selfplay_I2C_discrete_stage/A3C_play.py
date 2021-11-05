import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

from Design.Models.A3C_selfplay_I2C_discrete_stage.model import AC, FE
import Design.Environments.discrete_stage_creator_unwrapped as sc

np.set_printoptions(precision=3)
PATH = "./Design/Models/A3C_selfplay_I2C_discrete_stage/"
RES = 84

def density_vis(probs, env):
    n,m = len(sc.ANGLES), len(sc.DIAMS)
    img = probs.reshape(n,m)
    
    env.ax[2].clear()
    im = env.ax[2].imshow(img, cmap="YlGn")
    for i in range(n):
        for j in range(m):
            c = "w" if probs[i*m + j]>0.2 else "k"
            text = env.ax[2].text(j, i, np.round(probs[i*m + j],2), ha="center", va="center", color=c)
    # Turn spines off and create white grid.
    for edge, spine in env.ax[2].spines.items():
        spine.set_visible(False)

    env.ax[2].set_xticks(np.arange(m))
    env.ax[2].set_yticks(np.arange(n))
    env.ax[2].set_xticks(sc.DIAMS, minor=True)
    env.ax[2].set_yticks(sc.ANGLES, minor=True)
    # env.ax[2].grid(which="minor", color="w", linestyle='-', linewidth=3)
    env.ax[2].tick_params(which="minor", bottom=False, left=False)
    env.ax[2].set_ylabel("Angle $\phi$", fontsize=12)
    env.ax[2].set_xlabel("Diameter $d$", fontsize=12, labelpad=12)
    env.ax[2].xaxis.set_label_position('top')

    # Let the horizontal axes labeling appear on top.
    env.ax[2].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-j", "--job", default=0)
    args = parser.parse_args()
    device = torch.device("cpu")

    # Test env
    env = sc.StageCreator(RES, seed=args.seed, mode="selfplay")
    screen_shape = obs_size = env.observation_space["observation"]["screen"].shape
    state_size = 7 # x, y, i, d, x_target, y_target, i_target

    # Networks
    fe = FE(screen_shape, state_size).float()
    alice = AC(len(sc.DIAMS), len(sc.ANGLES), fe).float()
    bob = AC(len(sc.DIAMS), len(sc.ANGLES), fe).float()
    # alice.load_state_dict(torch.load(PATH+f"Alice-{args.job}.dat", map_location=torch.device(device)))
    # bob.load_state_dict(torch.load(PATH+f"Bob-{args.job}.dat", map_location=torch.device(device)))
    alice.load_state_dict(torch.load(PATH+f"Alice_last_4.dat", map_location=torch.device(device)))
    bob.load_state_dict(torch.load(PATH+f"Bob_last_4.dat", map_location=torch.device(device)))

    delay = .5
    
    episode = 0
    while episode<5:
        episode += 1
        a_steps, b_steps = 0, 0

        # Alice
        screen, state = env.reset(target=False)["observation"]
        state = np.hstack((state, env.goal))
        env.render(delay=delay)
        print(f"Alice, ep {episode}")
        done = False
        while not done: # if MAX_STEPS or MAX_TRAJ_LEN reached
            a_steps += 1
            state_t = torch.FloatTensor([state])
            screen_t = torch.FloatTensor([screen])

            # Act
            logits, _, stop_scores = alice(screen_t, state_t)
            probs = F.softmax(logits, dim=1)[0].detach().numpy()
            stop_probs = F.softmax(stop_scores, dim=1)[0].detach().numpy()
            action_idx = np.argmax(probs)
            stop = np.argmax(stop_probs)
            action = [action_idx, stop]
            phi = sc.ANGLES[action_idx//len(sc.DIAMS)]
            d = sc.DIAMS[action_idx%len(sc.DIAMS)]
            print(f"d={d}, phi={phi}, stop={stop}")

            if stop:
                if len(env.traj)>1:
                    print(f"Alice stopped after {len(env.traj)} gears, {a_steps} steps")
                    break
                else:
                    # pretend no stop has been played if the trajectory is too short
                    action[1]=0 

            # Perform step +[1] to indicate that it's alice
            obs, _, done, _ = env.step(action+[1])
            screen, state = obs["observation"]
            state = np.hstack((state, env.goal))
            # print(state)
            if len(env.traj)>env.old_len:
                density_vis(probs, env)
            env.render(delay=delay)
        
        print("Bob's turn")
        done=False
        env.set_target(state[:2], state[2])
        screen, state = env.reset(random=False)["observation"]
        state = np.hstack((state, env.goal))
        env.render(delay=delay)

        # Bob
        while not (done or a_steps+b_steps>sc.MAX_STEPS):
            b_steps += 1
            state_t = torch.FloatTensor([state])
            screen_t = torch.FloatTensor([screen])

            # Act
            logits, _, _ = bob(screen_t, state_t) 
            probs = F.softmax(logits, dim=1)[0].detach().numpy()
            action = [np.argmax(probs)]
            phi = sc.ANGLES[action[0]//len(sc.DIAMS)]
            d = sc.DIAMS[action[0]%len(sc.DIAMS)]
            print(f"d={d}, phi={phi}")

            # Perform step +[0] to indicate that it's bob
            obs, reward, done, _ = env.step(action+[0])
            screen, state = obs["observation"]
            state = np.hstack((state, env.goal))
            # print(state)
            if len(env.traj)>env.old_len:
                density_vis(probs, env)
            env.render(delay=delay)
            # if done and reward==0: # bob could have just occluded the output, not solve the env
            #     b_steps = sc.MAX_STEPS-a_steps
            #     # print(f"Bob solved an environment with {len(self.env.traj)} gears")

        print(f"Episode: {episode} | a_steps: {a_steps}, b_steps: {b_steps}, reward: {reward}")
