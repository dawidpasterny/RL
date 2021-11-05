import gym
import os
import sys

from torch.types import Device
sys.path.append(os.getcwd())

import numpy as np
import argparse
import datetime
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from common import SelfplayAgent, Tracker
from model import AC, FE
from Design.Models.AE import model as ae
import Design.Environments.discrete_stage_creator_unwrapped as sc

GAMMA = 0.98
LEARNING_RATE = 0.002
ENTROPY_BETA = 0.02

BATCH_SIZE = 128 # for NN training (only a min, the actual batch size may differ)
REPORT_INTERV = 100
TEST_INTERV = 1000
RES = 84
PATH = "./Design/Models/A3C_selfplay_I2C_discrete_stage/"

torch.multiprocessing.set_sharing_strategy('file_system')


def kernel(alice, bob, train_queue, device, seed=None):
    """ Execute by each single thread """
    env = sc.StageCreator(RES, mode="selfplay", boundary=0, seed=seed)
    agent = SelfplayAgent(alice, bob, env, device=device, gamma=GAMMA)

    while True:
        for _ in range(REPORT_INTERV):
            train_queue.put(agent.selfplay_episode(random=True))
        # Calculate rewards and send them to master for logging
        a_data, b_data = agent.selfplay_episode(random=True)
        mean_r_a, mean_r_b = agent.get_selfplay_rewards()
        train_queue.put((a_data, b_data, mean_r_a, mean_r_b, mp.current_process().name))


def cat_data(data:list, device):
    """ Returns a tuple of tensors"""
    screens = torch.cat(data[0]).to(device)
    states = torch.cat(data[1]).to(device)
    actions = torch.cat(data[2]).to(device)
    rewards = torch.cat(data[3]).to(device)

    return screens, states, actions, rewards


def train(train_data, net, optimizer, tracker, player, episode):
    screens, states, actions, rewards = train_data
    actions = actions.to(torch.int64)

    optimizer.zero_grad()
    logits, values, stop_logits = net(screens, states)

    value_loss = F.mse_loss(values.squeeze(-1), rewards) # since rewards come from a MC-like process
    adv = rewards.unsqueeze(dim=-1) - values.detach() # advantage (rewards are empirical Q)

    log_distr = F.log_softmax(logits, dim=1)
    # distr = F.softmax(logits-torch.max(logits, dim=1, keepdim=True)[0], dim=1) # -max for better stability
    distr = F.softmax(logits, dim=1) # -max for better stability
 
    # log probability of actions taken
    log_prob = log_distr.gather(1, actions[:,0].unsqueeze(-1)).squeeze(-1)
    if player=='alice':
        stop_distr = F.softmax(stop_logits, dim=1)
        stop_log_distr = F.log_softmax(stop_logits, dim=1)
        # log prob for binary stop action
        log_prob += stop_log_distr.gather(1, actions[:,-1].unsqueeze(-1)).squeeze(-1) 
        # For entropy regularization


    # Policy gradient loss
    policy_loss = -(adv*log_prob).mean()

    # Entropy loss
    ent = torch.sum(distr*log_distr, dim=1)
    if player=='alice':
        ent += torch.sum(stop_distr*stop_log_distr, dim=1) 
    entropy_loss = ENTROPY_BETA*ent.mean()

    loss = policy_loss + entropy_loss + value_loss
    loss.backward()
    optimizer.step()

    # Tracking
    tracker.track(f"{player} advantage", adv.mean(), episode)
    tracker.track(f"{player} values", values.mean(), episode)
    tracker.track(f"{player} entropy_loss", entropy_loss, episode)
    tracker.track(f"{player} policy_loss", policy_loss, episode)
    tracker.track(f"{player} value_loss", value_loss, episode)
    tracker.track(f"{player} loss_total", loss, episode)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    # num_threads = mp.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job",  default=0)
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-n", "--num_threads", default=mp.cpu_count()-2)
    parser.add_argument("-d", "--device", default="cpu")
    args = parser.parse_args()
    device = args.device
    num_threads = int(args.num_threads)
    writer = SummaryWriter(log_dir=PATH+"runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))
    
    # Test env
    test_env = sc.StageCreator(RES, seed=args.seed)
    screen_shape = obs_size = test_env.observation_space["observation"]["screen"].shape
    state_size = 7 # x, y, i, d, x_target, y_target, i_target

    # Networks
    fe = FE(screen_shape, state_size).share_memory().to(device).float()
    # fe.load_state_dict(torch.load(PATH+f"FE_best.dat", map_location=torch.device(device)))
    alice = AC(len(sc.DIAMS), len(sc.ANGLES), fe).share_memory().to(device).float()
    bob = AC(len(sc.DIAMS), len(sc.ANGLES), fe).share_memory().to(device).float()
    alice.load_state_dict(torch.load(PATH+f"Alice_last_4.dat", map_location=torch.device(device)))
    bob.load_state_dict(torch.load(PATH+f"Bob_last_4.dat", map_location=torch.device(device)))

    alice_opt = optim.Adam(alice.parameters(), lr=LEARNING_RATE, eps=1e-3)
    bob_opt = optim.Adam(bob.parameters(), lr=LEARNING_RATE, eps=1e-3)
    print(alice)
    print(bob)
    # [print(param.shape) for param in alice.parameters()]

    # Test agent
    test_agent = SelfplayAgent(alice, bob, test_env, device=device, gamma=GAMMA)
     
    # Schedule workers
    train_queue = mp.Queue(maxsize=num_threads)
    data_proc_list = []
    for n in range(num_threads):
        # Each agent gets different seed
        p = mp.Process(target=kernel, args=(alice, bob, train_queue, device))
        p.start()
        data_proc_list.append(p)

    a_data = [[],[],[],[]] # buffer for screens, states, actions, rewards
    b_data = [[],[],[],[]]
    batch_size = 0 # number of current entries in the buffers
    test_count = 0
    episode_count = 0
    best_test_reward = None
    best_mean_reward = -sc.MAX_STEPS
    best_ae_loss = None

    # Training loop
    try:
        with Tracker(writer,10) as tracker:
            while True:
                train_entry = train_queue.get()
                episode_count += 1

                # If train entry contains reward statistics
                if len(train_entry)>2:
                    mean_r_a, mean_r_b, thread = train_entry[-3:]
                    tracker.track("Alice's reward", mean_r_a, episode_count)
                    tracker.track("Bob's reward", mean_r_b, episode_count)
                    if best_mean_reward is None or best_mean_reward < mean_r_b:
                        print(f"Thread {thread}: best reward updated -> {mean_r_b:.3f}")
                        # torch.save(fe.state_dict(), PATH+f"A3C_selfplay_stage/FE-{args.job}.dat")
                        torch.save(alice.state_dict(), PATH+f"Alice-{args.job}.dat")
                        torch.save(bob.state_dict(), PATH+f"Bob-{args.job}.dat") 
                        best_mean_reward = mean_r_b                    
                    
                [a_data[i].append(t) for i,t in enumerate(train_entry[0])]
                [b_data[i].append(t) for i,t in enumerate(train_entry[1])]
                batch_size += len(b_data[-1][-1]) + len(a_data[-1][-1])

                if batch_size < BATCH_SIZE:
                    continue

                # Train Alice
                a_train_data = cat_data(a_data, device)
                train(a_train_data, alice, alice_opt, tracker, "alice", episode_count)

                # Train Bob if there are any data
                b_train_data = cat_data(b_data, device)
                if len(b_train_data[-1]) > 1:
                    train(b_train_data, bob, bob_opt, tracker, "bob", episode_count)
                
                # Clean up buffers (we are on-policy)
                a_data = [[],[],[],[]]
                b_data = [[],[],[],[]]
                batch_size = 0
            
            # t = steps_count//TEST_INTERV
            # if t > test_count:
            #     test_count = t
            #     mean_reward, mean_steps = agent.test_bob(10)
            #     print(f"\nJOB {args.job}, it {steps_count}: mean reward {mean_reward:.3f}, mean steps {mean_steps:.2f}")

            #     writer.add_scalar("Test_mean_reward_10", mean_reward, exp_count)
            #     writer.add_scalar("Test_mean_steps_10", mean_steps, exp_count)

            #     if best_test_reward is None or best_test_reward < mean_reward:
            #         if best_test_reward is not None:
            #             print(f"JOB {args.job}: best reward updated -> {mean_reward:.3f}")
            #             torch.save(fe.state_dict(), f"./FE-best-{args.job}.dat")
            #             torch.save(bob.state_dict(), f"./Bob_best-{args.job}.dat")
            #         best_test_reward = mean_reward

    finally: # triggers after ctr+C as well
        # Save the most current NN states
        torch.save(fe.state_dict(), PATH+f"FE_last_{args.job}.dat")
        torch.save(alice.state_dict(), PATH+f"Alice_last_{args.job}.dat")
        torch.save(bob.state_dict(), PATH+f"Bob_last_{args.job}.dat") 

        # Terminate all threads
        for p in data_proc_list:
            p.terminate()
            p.join()
