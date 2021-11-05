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
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as distr
import torch.nn.functional as F
import torch.multiprocessing as mp
# torch.autograd.set_detect_anomaly(True)

from Design.Models.SAC2_HER_stage_conv import model, common
# import Design.Environments.stage_creator_unwrapped as sc
import Design.Environments.discrete_stage_creator_unwrapped as sc


RES = 64
PATH = "./Design/Models/"

GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 100000
TEST_INTERV = 5000
UNROLL = 2
ENTROPY_ALPHA = 0.1
# CLIP_GRAD = 10

@torch.no_grad()
def test(act_net, env, num_tests=5, device="cpu"):
    """ Plays a number of episodes using actor net
        Returns average episode reward and step count
    """
    rewards = 0.0
    steps = 0
    for _ in range(num_tests):
        screen, obs = env.reset(random=False)["observation"]
        target = env.goal
        obs = np.hstack((obs, target))
        print("Init state: ", obs)
        while True: # play a full episode
            state_t = torch.FloatTensor(np.array([obs])).to(device)
            screen_t = torch.FloatTensor(np.array([screen])).to(device)
            mu, _ = act_net(screen_t, state_t) # just take the expected action
            mu = mu.detach().numpy()[0]
            action = [sc.DIAMS[min(max(2,round(mu[0]/0.05)),5)-2], sc.ANGLES[min(round(mu[1]/(np.pi/3)),5)]] 
            print(f"d={mu[0]:.2f}, phi={mu[1]:.2f}")
            print(f"action[0]={action[0]}, action[1]={action[1]}")

            new_obs, reward, done, _ = env.step(action)
            obs = np.hstack((new_obs["observation"][1], target))
            screen = new_obs["observation"][0]
            print("Next state: ", obs)

            rewards += reward
            steps += 1
            if done:
                break
    return rewards/num_tests, steps/num_tests


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job",  default=0)
    parser.add_argument("-n", "--num_threads", default=mp.cpu_count()/2-1)
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-d", "--device", default="cpu")
    args = parser.parse_args()

    job = args.job
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    writer = SummaryWriter(log_dir=PATH+"SAC2_HER_stage_conv/runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S")+f"_job{job}")
    print(f"Executing job: {job} on {device}")

    # Env
    env = sc.StageCreator(RES, boundary=0, mode="goal", seed=seed) 
    env.mode = "selfplay" # so that collisions do not lead to termination
    test_env = sc.StageCreator(RES, boundary=0, seed=seed, mode="goal")

    screen_shape = obs_size = test_env.observation_space["observation"]["screen"].shape
    state_size = 7 # x, y, i, d, x_target, y_target, i_target
    act_size = 2

    # Networks
    fe = model.FE(screen_shape, state_size).share_memory()
    act_net = model.ActorSAC(act_size, fe).to(device).share_memory()
    crt_net = model.CriticSAC(act_size, fe).to(device)
    tgt_crt_net = model.TargetNet(crt_net)

    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    print(fe)
    print(act_net)
    print(crt_net)
    print(f"Starting job #{job} on {args.device}")
    
    # Agent and buffer
    buffer = common.ExperienceBuffer(buffer_size=REPLAY_SIZE, device=device) # indiovidual exp replay
    agent = common.Agent_SAC_HER(act_net, env, buffer, GAMMA, device=device, unroll_steps=UNROLL)

    best_test_reward = None
    exp_count = 0
    test_count = 0
    try:
        with common.Tracker(writer,10) as tracker:
            while True:
                exp_count+= agent.play_episode(random=True)

                if exp_count%20==0:
                    mean_episode_reward, mean_step_count = agent.get_mean_reward_and_steps()
                    writer.add_scalar("Episode_reward_100", mean_episode_reward, exp_count)
                    writer.add_scalar("Episode_steps_100", mean_step_count, exp_count)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                # obs is the concatenated current state and target (len=7)
                screens, obs, acts, rewards, dones, next_screens, next_obs = buffer.sample(BATCH_SIZE)
                
                # Double-Q(s,a)
                q1, q2 = crt_net(screens, obs, acts)
                q_min = torch.min(q1,q2).squeeze()
                tracker.track("Q(s,a)_batch_mean", q_min.mean(), exp_count)

                # Double-Q(s',a') 
                next_mu, next_std = act_net(next_screens, next_obs) # no action cliping here
                next_q1, next_q2 = tgt_crt_net(next_screens, next_obs, next_mu) # take expected action, what else?

                # Q_ref
                next_acts_distr = distr.Normal(next_mu, next_std)
                v_next = torch.min(next_q1, next_q2).squeeze() - \
                    ENTROPY_ALPHA*next_acts_distr.log_prob(next_mu).sum(dim=1) # eqn (3) in 2nd SAC paper

                v_next[dones] = 0.0 # if terminated, local reward makes up the value already
                q_ref = (rewards + v_next*(GAMMA**UNROLL)).unsqueeze(dim=-1)
                tracker.track("Q_ref(s,a)_batch_mean", q_ref.mean(), exp_count)

                # Critic loss (on both Double-Q heads to update all parameters)
                q1_loss = F.mse_loss(q1, q_ref.detach()) # eqn (5) in 2nd SAC paper
                q2_loss = F.mse_loss(q2, q_ref.detach())
                q_loss = q1_loss + q2_loss
                q_loss.backward(retain_graph=True)
                crt_opt.step()
                tracker.track("Loss_crt", q_loss, exp_count)

                # Actor loss
                # MC empirical action entropy
                act_opt.zero_grad()
                mu, std = act_net(screens, obs)
                acts_distr = distr.Normal(mu, std)
                actions = acts_distr.sample()
                act_ent = ENTROPY_ALPHA * acts_distr.log_prob(actions).sum(dim=1) 
                tracker.track("Entropy_act_batch", -act_ent.mean(), exp_count)

                q_out, _ = crt_net(screens, obs, actions)
                # Actors job is to find actions that maximize Q
                act_loss = (act_ent - q_out).mean() # eqn (9) in 2nd SAC paper
                act_loss.backward()
                # nn.utils.clip_grad_norm_(act_net.parameters(), CLIP_GRAD) # to awoid blow-up
                act_opt.step() # only actors optimizer, don't touch q networks
                tracker.track("Loss_act", act_loss, exp_count)

                # Soft sync of the target network
                tgt_crt_net.alpha_sync(alpha = 1 -1e-3)


                # Test
                t = exp_count//TEST_INTERV
                if t > test_count:
                    test_count = t
                    mean_reward, mean_steps = test(act_net, test_env, device=device)
                    print(f"\nJOB {job}, it {exp_count}: mean reward {mean_reward:.3f}, mean steps {mean_steps:.2f}")

                    writer.add_scalar("Test_mean_reward_10", mean_reward, exp_count)
                    writer.add_scalar("Test_mean_steps_10", mean_steps, exp_count)

                    if best_test_reward is None or best_test_reward < mean_reward:
                        if best_test_reward is not None:
                            print(f"JOB {job}: best reward updated -> {mean_reward:.3f}")
                            # torch.save(fe.state_dict(), PATH + f"DDPG_HER_stage/FE-best-{job}.dat")
                            torch.save(act_net.state_dict(), PATH + f"SAC2_HER_stage_conv/Actor_best-{job}.dat")
                            torch.save(crt_net.state_dict(), PATH + f"SAC2_HER_stage_conv/Critic_best-{job}.dat")
                        best_test_reward = mean_reward

    finally:
        torch.save(act_net.state_dict(), PATH + f"SAC2_HER_stage_conv/Actor_last-{job}.dat")
        torch.save(crt_net.state_dict(), PATH + f"SAC2_HER_stage_conv/Critic_last-{job}.dat")
