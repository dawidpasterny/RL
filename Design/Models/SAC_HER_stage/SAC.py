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
import torch.distributions as distr
import torch.nn.functional as F
import torch.multiprocessing as mp

from Design.Models.SAC_HER_stage import model, common
from Design.Models.AE.model import Autoencoder84
import Design.Environments.stage_creator_unwrapped as sc


RES = 84
PATH = "./Design/Models/"

GAMMA = 0.99
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 100
TEST_INTERV = 500
UNROLL = 2
ENTROPY_ALPHA = 0.1
CLIP_GRAD = 10

@torch.no_grad()
def test(net, env, num_tests=5, device="cpu", random=True):
    """ Plays a number of episodes using actor net
        Returns average episode reward and step count
    """
    rewards = 0.0
    steps = 0
    for _ in range(num_tests):
        obs = env.reset(random=random)["observation"]
        target = env.goal
        obs = np.hstack((obs, target))
        print("Init state: ", obs[-7:])
        while True: # play a full episode
            obs_t = torch.FloatTensor([obs]).to(device)
            # the reason not to use agent here is to just follow the policy
            # we don't need exploration (hence no clipping too)
            mu, _ = net(obs_t) # mean of the action distr
            action = mu.cpu().numpy()[0]
            print("Action: ", action)
            new_obs, reward, done, _ = env.step(action)
            obs = np.hstack((new_obs["observation"], target))
            print("Next state: ", obs[-7:])
            rewards += reward
            steps += 1
            if done:
                break
    return rewards/num_tests, steps/num_tests


def kernel(act_net, train_queue, device, num_threads, env_seed):

    ae = Autoencoder84(1, pretrained="./Design/Models/AE/Autoencoder84.dat").to(device).eval()
    env = sc.StageCreator(RES, ae=ae, boundary=.8, mode="goal", seed=env_seed) # individual environment
    env.mode = "selfplay"
    buffer = common.ExperienceBuffer(buffer_size=REPLAY_SIZE//num_threads,device=device) # indiovidual exp replay
    agent = common.Agent_SAC_HER(act_net, env, buffer, GAMMA, device=device, unroll_steps=UNROLL)
    thread_id = mp.current_process().name
    exp_count=0
    
    while True:
        exp_count=agent.play_episode(random=False) # play same episode over and over again
        mean_episode_reward, mean_step_count = agent.get_mean_reward_and_steps()

        if len(buffer) < REPLAY_INITIAL//num_threads:
            continue

        batch = buffer.sample(BATCH_SIZE)
        train_queue.put((batch, exp_count, mean_episode_reward, mean_step_count, thread_id))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-j", "--job",  required=True)
    # parser.add_argument("-n", "--num_threads", required=True)
    # parser.add_argument("-s", "--seed", default=None)
    # parser.add_argument("-d", "--device")
    # args = parser.parse_args()

    
    # num_threads = int(args.num_threads)
    # job = args.job
    # seed = int(args.seed)
    job = 0
    num_threads = 3
    seed = 12345678
    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    device = torch.device("cpu")
    ae = Autoencoder84(1, pretrained="./Design/Models/AE/Autoencoder84.dat", device=device).eval()

    writer = SummaryWriter(log_dir=PATH+"SAC_HER_stage/runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S")+f"_job{job}")
    print(f"Executing job: {job} on {device}")

    # Env
    test_env = sc.StageCreator(RES, ae=ae, boundary=.8, seed=seed)
    obs_size = test_env.observation_space["observation"].shape[0] + \
                test_env.observation_space["desired_goal"].shape[0]
    act_size = test_env.action_space.shape[0]

    # Networks
    act_net = model.ModelActor(obs_size, act_size).to(device).share_memory()
    crt_net = model.ModelCritic(obs_size).to(device)
    twinq_net = model.ModelSACTwinQ(obs_size, act_size).to(device)

    # act_net.load_state_dict(torch.load(PATH+f"SAC_HER_stage/Actor_best-{job}.dat", map_location=torch.device(device)))
    # crt_net.load_state_dict(torch.load(PATH+f"SAC_HER_stage/Critic_best-{job}.dat", map_location=torch.device(device)))
    # twinq_net.load_state_dict(torch.load(PATH+f"SAC_HER_stage/Twinq_best-{job}.dat", map_location=torch.device(device)))
    tgt_crt_net = model.TargetNet(crt_net)

    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    twinq_opt = optim.Adam(twinq_net.parameters(), lr=LEARNING_RATE)

    print(act_net)
    print(crt_net)
    print(twinq_net)
    print(f"Starting job #{job}")
    
    
    with torch.no_grad():
        # Initialize multiple agents
        train_queue = mp.Queue(maxsize=2*num_threads)
        data_proc_list = []
        for _ in range(num_threads):
            p = mp.Process(target=kernel, args=(act_net, train_queue, device, num_threads, seed))
            p.start()
            data_proc_list.append(p)

    best_test_reward = None
    exp_count = 0
    test_count = 0
    batch = []
    k = 0
    try:
        with common.Tracker(writer,10) as tracker:
            while True:
                # Gather learner's batch from workers
                worker_batch, worker_count, mean_episode_reward, mean_step_count, thread_id = train_queue.get()
                writer.add_scalar(f"Episode_reward_100_{thread_id}", mean_episode_reward, exp_count)
                writer.add_scalar(f"Episode_steps_100_{thread_id}", mean_step_count, exp_count)
                exp_count += worker_count
                k += 1
                if batch == []:
                    for entry in worker_batch:
                        batch.append(entry)
                else:
                    for i, entry in enumerate(worker_batch):
                        batch[i] = torch.cat((batch[i], entry))
                
                if k<num_threads:
                    continue

                obs, actions, rewards, dones, next_obs = batch
                batch = []
                k = 0
                
                # Preprocess batch
                v_next = tgt_crt_net(next_obs)
                v_next[dones] = 0.0 # if terminated, local reward makes up the value already
                q_ref = rewards.unsqueeze(dim=-1) + v_next*GAMMA**UNROLL # eqn (8) in SAC paper

                mu, std = act_net(obs)
                print(f"mu: {mu[0]}, std: {std}")
                acts_distr = distr.Normal(mu, std)
                acts = acts_distr.sample()
                q1, q2 = twinq_net(obs, acts)
                v_ref = torch.min(q1, q2).squeeze() - \
                        ENTROPY_ALPHA*acts_distr.log_prob(acts).sum(dim=1) # eqn (3) in SAC paper

                tracker.track("V_ref_batch_mean", v_ref.mean(), exp_count)
                tracker.track("Q_ref_batch_mean", q_ref.mean(), exp_count)

                # Twin Q
                twinq_opt.zero_grad()
                q1, q2 = twinq_net(obs, actions)
                q1_loss = F.mse_loss(q1, q_ref.detach()) # eqn (7) in SAC paper
                q2_loss = F.mse_loss(q2, q_ref.detach())
                q_loss = q1_loss + q2_loss
                q_loss.backward()
                twinq_opt.step()
                tracker.track("Loss_double_Q", q_loss, exp_count)

                # Critic
                crt_opt.zero_grad()
                v = crt_net(obs)
                v_loss = F.mse_loss(v.squeeze(), v_ref.detach()) # eqn (5) in SAC paper
                v_loss.backward()
                crt_opt.step()
                tracker.track("Loss_V", v_loss, exp_count)

                # Actor
                act_opt.zero_grad()
                q_out, _ = twinq_net(obs, mu) # take the mean mu as an action
                act_ent = ENTROPY_ALPHA * acts_distr.log_prob(mu).sum(dim=1)
                tracker.track("Entropy_act_batch_mean", act_ent.mean(), exp_count)
                tracker.track("Actor_Q_batch_mean", q_out.mean(), exp_count)

                # act_loss = -q_out.mean() # improper loss (without entropy)
                act_loss = (act_ent - q_out).mean() # eqn (4), (10), (12)
                act_loss.backward()
                nn.utils.clip_grad_norm_(act_net.parameters(), CLIP_GRAD) # to awoid blow-up
                act_opt.step() # only actors optimizer, don't touch q networks
                tracker.track("Loss_act", act_loss, exp_count)

                # Soft sync of the target network
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)


                # Test
                t = exp_count//TEST_INTERV
                if t > test_count:
                    test_count = t
                    mean_reward, mean_steps = test(act_net, test_env, device=device, random=False)
                    print(f"\nJOB {job}, it {exp_count}: mean reward {mean_reward:.3f}, mean steps {mean_steps:.2f}\n\n")

                    writer.add_scalar("Test_mean_reward_10", mean_reward, exp_count)
                    writer.add_scalar("Test_mean_steps_10", mean_steps, exp_count)

                    if best_test_reward is None or best_test_reward < mean_reward:
                        if best_test_reward is not None:
                            print(f"JOB {job}: best reward updated -> {mean_reward:.3f}")
                            # torch.save(fe.state_dict(), PATH + f"DDPG_HER_stage/FE-best-{job}.dat")
                            torch.save(act_net.state_dict(), PATH + f"SAC_HER_stage/Actor_best-{job}.dat")
                            torch.save(crt_net.state_dict(), PATH + f"SAC_HER_stage/Critic_best-{job}.dat")
                            torch.save(twinq_net.state_dict(), PATH + f"SAC_HER_stage/Twinq_best-{job}.dat")
                        best_test_reward = mean_reward

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
