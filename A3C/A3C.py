import gym
import os
import sys
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

from common import SelfplayAgent, Tracker, MAX_STEPS
from model import AC, FE, AE
import stage_creator as sc

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1

# NUM_THREADS = 8 
NUM_FEATURES = 128
BATCH_SIZE = 32
TEST_INTERV = 1000
RES = 84
PATH = "./"

torch.multiprocessing.set_sharing_strategy('file_system')


def kernel(alice, bob, train_queue, device):
    """ Execute by each single thread """
    env = sc.StageCreator()
    env = sc.ScreenOutput(RES, env)
    agent = SelfplayAgent(alice, bob, env, device=device, gamma=GAMMA)

    while True:
        for _ in range(10):
            train_queue.put(agent.selfplay_episode())
        a_data, b_data = agent.selfplay_episode()
        mean_r_a, mean_r_b = agent.get_selfplay_rewards()
        train_queue.put((a_data, b_data, mean_r_a, mean_r_b, mp.current_process().name))


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def cat_data(data:list):
    """ Returns a tuple of tensors"""
    screens = torch.cat(data[0])
    states = torch.cat(data[1])
    actions = torch.cat(data[2])
    rewards = torch.cat(data[3])

    return screens, states, actions, rewards


def sample_ae_batch(screens, batch_size):
    indices = np.random.choice(len(screens), batch_size, replace=False)
    return screens[indices]
    

def train(train_data, net, optimizer, tracker, name):
    screens, states, actions, rewards = train_data

    optimizer.zero_grad()
    mu, var, values, p = net(screens, states)

    value_loss = F.mse_loss(values.squeeze(-1), rewards)

    adv = rewards.unsqueeze(dim=-1) - values.detach()
    log_prob = adv*calc_logprob(mu, var, actions)
    if name=='alice':
        log_prob = torch.cat([log_prob, torch.log(p)], dim=1)

    policy_loss = -log_prob.mean()
    ent = -(torch.log(2*math.pi*var) + 1)/2
    entropy_loss = ENTROPY_BETA*ent.mean()

    loss = policy_loss + entropy_loss + value_loss
    loss.backward()
    optimizer.step()

    # Tracking
    tracker.track(f"{name} advantage", adv, steps_count)
    tracker.track(f"{name} values", values, steps_count)
    tracker.track(f"{name} entropy_loss", entropy_loss, steps_count)
    tracker.track(f"{name} policy_loss", policy_loss, steps_count)
    tracker.track(f"{name} value_loss", value_loss, steps_count)
    tracker.track(f"{name} loss_total", loss, steps_count)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    # num_threads = mp.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job",  default=0)
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-n", "--num_threads", default=mp.cpu_count())
    args = parser.parse_args()
    device = "cpu"
    #num_threads = int(args.num_threads)-1
    num_threads = 32

    writer = SummaryWriter(log_dir=PATH+"/runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))
    fe = FE((1,RES,RES), NUM_FEATURES).share_memory().float()
    ae = AE(fe).float()
    ae_opt = optim.Adam(ae.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    alice = AC(NUM_FEATURES+6, 2, fe).share_memory().float()
    bob = AC(NUM_FEATURES+6, 2, fe).share_memory().float()
    alice_opt = optim.Adam(alice.parameters(), lr=LEARNING_RATE, eps=1e-3)
    bob_opt = optim.Adam(bob.parameters(), lr=LEARNING_RATE, eps=1e-3)
    print(alice)
    print(bob)
    print(ae)
    
    # #For testing only 
    # env = sc.StageCreator(seed=args.seed)
    # obs_size = env.observation_space.shape[0] 
    # agent = SelfplayAgent(alice, bob, env, device=device, gamma = GAMMA)
     
    train_queue = mp.Queue(maxsize=num_threads)
    data_proc_list = []
    for _ in range(num_threads):
        p = mp.Process(target=kernel, args=(alice, bob, train_queue, device))
        p.start()
        data_proc_list.append(p)

    a_data = [[],[],[],[]]
    b_data = [[],[],[],[]]
    steps_count = 0
    i = 0
    batch_size = 0
    test_count = 0
    best_test_reward = None
    best_mean_reward = -MAX_STEPS
    best_ae_loss = None

    try:
        with Tracker(writer,10) as tracker:
            while True:
                train_entry = train_queue.get()
                if len(train_entry)>2:
                    mean_r_a, mean_r_b, thread = train_entry[-3:]
                    tracker.track("Alice's reward", mean_r_a, steps_count)
                    tracker.track("Bob's reward", mean_r_b, steps_count)
                    if best_mean_reward is None or best_mean_reward < mean_r_b:
                        print(f"{thread}: best reward updated -> {mean_r_b:.3f}")
                        torch.save(fe.state_dict(), PATH+f"FE-{args.job}.dat")
                        torch.save(alice.state_dict(), PATH+f"Alice-{args.job}.dat")
                        torch.save(bob.state_dict(), PATH+f"Bob-{args.job}.dat") 
                        best_mean_reward = mean_r_b                    
                    
                [a_data[i].append(t) for i,t in enumerate(train_entry[0])]
                [b_data[i].append(t) for i,t in enumerate(train_entry[1])]
                steps_count += len(b_data[-1][-1]) + len(a_data[-1][-1])
                batch_size += len(b_data[-1][-1]) + len(a_data[-1][-1])

                if batch_size < BATCH_SIZE*num_threads:
                    continue

                # print(f'{mp.current_process().name} trains Bob and Alice')
                a_train_data = cat_data(a_data)
                train(a_train_data, alice, alice_opt, tracker, "alice")
                b_train_data = cat_data(b_data)
                if len(b_train_data[0]) != 1:
                    train(b_train_data, bob, bob_opt, tracker, "bob")
                
                # Autoencoder pass
                i += 1
                if i%10==0:
                    screens = torch.cat([a_train_data[0], b_train_data[0]])
                    training_batch = sample_ae_batch(screens, 128)
                    
                    ae_opt.zero_grad()
                    out = ae.forward(training_batch)
                    ae_loss = torch.nn.MSELoss()(out, training_batch)
                    ae_loss.backward()
                    ae_opt.step()

                    if best_ae_loss is None or ae_loss < best_ae_loss:
                        torch.save(ae.state_dict(), PATH + f"AE-{args.job}.dat")
                        #if best_ae_loss is not None:
                            #print(f"Best loss updated {best_loss} -> {loss}, model saved")
                        best_ae_loss = ae_loss

                    writer.add_scalar("AE loss", ae_loss, i)

                
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

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
