#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd())

import time
import datetime
import gym
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lib import model, common
from Design.Environments import stage_creator as sc

GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 80000
REPLAY_INITIAL = 8000
TEST_INTERV = 1000
UNROLL = 2


def test(net, ae, env, count=10, device="cpu"):
    """ Plays a number of episodes using actor net
        Returns average episode reward and step count
    """
    rewards = 0.0
    steps = 0
    for _ in range(count):
        screen, state = env.reset()
        while True: # play a full episode
            state_t = torch.tensor([state]).to(device).float()
            screen_t = torch.tensor([screen]).to(device)
            features = torch.reshape(ae.encode(screen_t), (1,-1)).float()
            # the reason not to use agent here is to just follow the policy
            # we don't need exploration
            action = net(torch.column_stack((features, state_t)))[0].data.cpu().numpy()
            (screen, state), reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards/count, steps/count



if __name__ == "__main__":
    device = torch.device("cpu")
    save_path = "./Design/Models/DDPG/"
    writer = SummaryWriter(log_dir="./Design/Models/DDPG/runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))

    # Envs
    env = sc.StageCreator(seed=3672871121734420758)
    obs_size = env.observation_space.shape[0]
    env = sc.ScreenOutput(64, env) # 100X100 grid for CNN
    test_env = sc.StageCreator()
    test_env = sc.ScreenOutput(64, test_env)

    # Networks
    ae = model.Autoencoder(1, pretrained="./Design/Models/DDPG/Autoencoder-FC.dat").float().to(device).float()
    # fe = lambda x: ae.encode(x)
    obs_size += 64
    act_net = model.DDPGActor(obs_size, env.action_space.shape[0]).to(device).float()
    crt_net = model.DDPGCritic(obs_size, env.action_space.shape[0]).to(device).float()
    
    tgt_act_net = model.TargetNet(act_net) # behavioral policy?
    tgt_crt_net = model.TargetNet(crt_net)
    print(act_net)
    print(crt_net)

    buffer = common.ExperienceBuffer(buffer_size=REPLAY_SIZE)
    agent = common.AgentDDPG(act_net, env, buffer, ae, GAMMA, device=device, \
                                        ou_epsilon=1.0, unroll_steps=UNROLL)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    ae_opt = optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    
    best_test_reward = None
    exp_count=0
    test_count=0
    ae_loss_best=0.05
    with common.Tracker(writer,10) as tracker:
        while True:
            exp_count+=agent.play_episode()
            mean_episode_reward, mean_step_count = agent.get_mean_reward_and_steps()
            writer.add_scalar("Episode_reward_100", mean_episode_reward, exp_count)
            writer.add_scalar("Episode_steps_100", mean_step_count, exp_count)

            if len(buffer) < REPLAY_INITIAL:
                continue

            screens, states, actions, rewards, dones, next_screens, next_states \
                                = buffer.sample(BATCH_SIZE)

            # Reduce dimensionality of the state screens
            state_features = ae.encode(screens)
            next_state_features = ae.encode(next_screens)

            # # Train auto encoder
            # ae_opt.zero_grad()
            # out = ae.decode(state_features)
            # ae_loss = nn.functional.mse_loss(out, screens)
            # ae_loss.backward()
            # ae_opt.step()
            # if ae_loss.item()<ae_loss_best:
            #     torch.save(ae.state_dict(), save_path + "Autoencoder_best_2.dat")

            # tracker.track("Loss_AE", ae_loss, exp_count)

            # Concatenate with states
            states = torch.column_stack((torch.reshape(state_features.detach(), (BATCH_SIZE,-1)),states))
            next_states = torch.column_stack((torch.reshape(next_state_features.detach(), (BATCH_SIZE,-1)),next_states))

            # Train critic
            crt_opt.zero_grad()
            q = crt_net(states, actions) # Q(s,a)
            next_act = tgt_act_net.target_model(next_states) 
            q_next = tgt_crt_net.target_model(next_states, next_act) # Q(s',a')
            q_next[dones] = 0.0 # if terminated, don't take Q but only local reward
            q_ref = rewards.unsqueeze(dim=-1) + q_next * GAMMA**UNROLL
            critic_loss = nn.functional.mse_loss(q, q_ref.detach()) # detach not to propagate to target NN
            critic_loss.backward()
            crt_opt.step()

            tracker.track("Loss_critic", critic_loss, exp_count)
            tracker.track("Q_ref", q_ref.mean(), exp_count)

            # Train actor
            act_opt.zero_grad()
            cur_act = act_net(states.detach()) # detach not to modify Q-val here
            # Adjust actor's actions to maximize critics output Q
            actor_loss = -crt_net(states, cur_act) # max = -min
            actor_loss = actor_loss.mean()
            actor_loss.backward() 
            act_opt.step()

            tracker.track("Loss_actor", actor_loss, exp_count)
            

            # Sync target networks
            tgt_act_net.alpha_sync(alpha=1 - 1e-3)
            tgt_crt_net.alpha_sync(alpha=1 - 1e-3)


            # Test
            t = exp_count//TEST_INTERV
            if t > test_count:
                test_count = t
                mean_reward, mean_steps = test(act_net, ae, test_env, device=device)
                print(f"Mean reward {mean_reward:.3f}, mean steps {mean_steps:.2f}")

                writer.add_scalar("Test_mean_reward_10", mean_reward, exp_count)
                writer.add_scalar("Test_mean_steps_10", mean_steps, exp_count)

                if best_test_reward is None or best_test_reward < mean_reward:
                    if best_test_reward is not None:
                        print(f"Best reward updated -> {mean_reward:.3f}")
                        torch.save(act_net.state_dict(), save_path + "Actor-best.dat")
                        torch.save(crt_net.state_dict(), save_path + "Critic_best.dat")
                        # torch.save(ae.state_dict(), save_path + "Autoencoder_best_2.dat")
                    best_test_reward = mean_reward

    