#!/usr/bin/env python3
import sys
import os
sys.path.append(os.getcwd())

import time
import datetime
import gym
import numpy as np
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Design.Models.DDPG_HER_stage.lib import model, common
from Design.Models.AE import model as ae
import Design.Environments.stage_creator as sc

GAMMA = 0.99
BATCH_SIZE = 512 # so big because of HER and because one iteration yields entire episode data (also because of HER)
LEARNING_RATE = 2e-4
REPLAY_SIZE = 80000
REPLAY_INITIAL = 8000
TEST_INTERV = 1000
UNROLL = 2

PATH = "./Design/Models/"

def test(net, env, num_tests=5, device="cpu"):
    """ Plays a number of episodes using actor net
        Returns average episode reward and step count
    """
    rewards = 0.0
    steps = 0
    for _ in range(num_tests):
        screen, state = env.reset()
        print("Init state: ", state)
        while True: # play a full episode
            state_t = torch.tensor([state]).to(device).float()
            screen_t = torch.tensor([screen]).to(device)
            # the reason not to use agent here is to just follow the policy
            # we don't need exploration (hence no clipping too)
            action = net(screen_t, state_t)[0].data.cpu().numpy()
            print("Action: ", action)
            (screen, state), reward, done, _ = env.step(action)
            print("Next state: ", state)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards/num_tests, steps/num_tests



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job",  required=True)
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-d", "--device")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"
    job = args.job
    seed = args.seed

    # device = "cpu"
    # job=0
    # seed=None

  
    writer = SummaryWriter(log_dir=PATH+"DDPG_HER_stage/runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))
    print(f"Executing job: {job} on {device}")

    # Envs
    env = sc.StageCreator(seed=seed, boundary=0.5)
    obs_size = env.observation_space.shape[0]
    env = sc.ScreenOutput(64, env)
    test_env = sc.StageCreator(boundary=0.5)
    test_env = sc.ScreenOutput(64, test_env)

    # Networks
    fe = ae.Autoencoder(1, pretrained=PATH + "AE/Autoencoder-FC.dat", device=device).to(device).float().eval() # eval locks the gradients
    # fe = model.FE((1,64,64), 128).to(device).float()
    obs_size += 64
    act_net = model.DDPGActor(obs_size, env.action_space.shape[0], fe).to(device).float()
    crt_net = model.DDPGCritic(obs_size, env.action_space.shape[0], fe).to(device).float()

    tgt_act_net = model.TargetNet(act_net) 
    tgt_crt_net = model.TargetNet(crt_net)
    print(f"Starting job #{job}")
    # print(fe)
    #print(act_net)
    #print(crt_net)

    buffer = common.ExperienceBuffer(buffer_size=REPLAY_SIZE,device=device)
    agent = common.AgentDDPG(act_net, env, buffer, GAMMA, device=device, ou_epsilon=0.25,  unroll_steps=UNROLL)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    
    best_test_reward = None
    exp_count=0
    test_count=0
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

            
            # Train critic
            crt_opt.zero_grad()
            q = crt_net(screens, states, actions) # Q(s,a)
            next_act = tgt_act_net.target_model(next_screens, next_states) 
            q_next = tgt_crt_net.target_model(next_screens, next_states, next_act) # Q(s',a')
            q_next[dones] = 0.0 # if terminated, don't take Q but only local reward
            q_ref = rewards.unsqueeze(dim=-1) + q_next * GAMMA**UNROLL
            critic_loss = nn.functional.mse_loss(q, q_ref.detach()) # detach not to propagate to target NN
            critic_loss.backward()
            crt_opt.step()

            tracker.track("Loss_critic", critic_loss, exp_count)
            tracker.track("Q_ref", q_ref.mean(), exp_count)

            # Train actor
            act_opt.zero_grad()
            cur_act = act_net(screens, states) # detach not to modify Q-val here
            # Adjust actor's actions to maximize critics output Q
            actor_loss = -crt_net(screens, states, cur_act) # actor loss is Q to be maxed! (max = -min)
            actor_loss = actor_loss.mean()
            actor_loss.backward() 
            act_opt.step()

            tracker.track("Loss_actor", actor_loss, exp_count)
            

            # Soft sync target networks
            tgt_act_net.alpha_sync(alpha=1 - 1e-3)
            tgt_crt_net.alpha_sync(alpha=1 - 1e-3)


            # Test
            t = exp_count//TEST_INTERV
            if t > test_count:
                test_count = t
                mean_reward, mean_steps = test(act_net, test_env, device=device)
                print(f"\nJOB {job}, it {exp_count}: mean reward {mean_reward:.3f}, mean steps {mean_steps:.2f}")

                writer.add_scalar("Test_mean_reward_10", mean_reward, exp_count)
                writer.add_scalar("Test_mean_steps_10", mean_steps, exp_count)

                # if test_count>300:
                #    torch.save(fe.state_dict(), PATH + f"DDPG_HER_stage/FE-worst-{job}.dat")
                #    torch.save(act_net.state_dict(), PATH + f"DDPG_HER_stage/Actor-worst-{job}.dat")
                #    torch.save(crt_net.state_dict(), PATH + f"DDPG_HER_stage/Critic_worst-{job}.dat")

                if best_test_reward is None or best_test_reward < mean_reward:
                    if best_test_reward is not None:
                        print(f"JOB {job}: best reward updated -> {mean_reward:.3f}")
                        # torch.save(fe.state_dict(), PATH + f"DDPG_HER_stage/FE-best-{job}.dat")
                        torch.save(act_net.state_dict(), PATH + f"DDPG_HER_stage/Actor-best-{job}.dat")
                        torch.save(crt_net.state_dict(), PATH + f"DDPG_HER_stage/Critic_best-{job}.dat")
                    best_test_reward = mean_reward

    
