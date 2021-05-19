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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib import model, common
from Design.Environments import stage_creator as sc

GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
TEST_ITERS = 1000



def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = torch.tensor(np.array([obs], dtype=np.float32)).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count



if __name__ == "__main__":
    device = torch.device("cpu")
    save_path = "./Design/Models/DDPG/"
    writer = SummaryWriter(log_dir="./Design/Models/DDPG/runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))

    # Envs
    env = sc.StageCreator(seed=3672871121734420758)
    obs_size = env.observation_space.shape[0]
    env = sc.ScreenOutput(100, env) # 100X100 grid for CNN
    test_env = sc.StageCreator(seed=3672871121734420758)
    test_env = sc.ScreenOutput(100, test_env)

    # Networks
    fe = model.FeatureExtractor(1, pretrained=True).float().to(device)
    obs_size += fe.get_out_size()
    act_net = model.DDPGActor(obs_size, env.action_space.shape[0]).to(device)
    crt_net = model.DDPGCritic(obs_size, env.action_space.shape[0]).to(device)
    
    tgt_act_net = model.TargetNet(act_net) # behavioral policy?
    tgt_crt_net = model.TargetNet(crt_net)
    print(act_net)
    print(crt_net)

    agent = model.AgentDDPG(act_net, device=device)
    exp_source = common.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = common.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    with common.RewardTracker(writer) as tracker:
        with common.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v \
                                        = common.unpack_batch(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v) # Q(s,a)
                last_act_v = tgt_act_net.target_model(last_states_v) 
                q_next_v = tgt_crt_net.target_model(last_states_v, last_act_v) # Q(s',a')
                q_next_v[dones_mask] = 0.0 # if terminated, don't take Q but only reward
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_next_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach()) # detach not to propagate to target NN
                critic_loss_v.backward()
                crt_opt.step()

                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                # Adjust actors actions to maximize critics output Q
                actor_loss_v = -crt_net(states_v, cur_actions_v) # max = -min
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward() # only actor's optimizer, don't modify Q here
                act_opt.step()

                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # Sync target networks
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                # Test
                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print(f"Test done in {time.time()-ts:.2f} sec, reward {reward:.3f}, steps {steps}}")

                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)

                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(f"Best reward updated -> {rewards:.3f}")
                            torch.save(act_net.state_dict(), save_path + "Actor-best.dat")
                        best_reward = rewards

    torch.save(crt_net.state_dict(), save_path + "Critic_"+datetime.datetime.now().strftime("%b%d_%H_%M_%S")+".dat")
    torch.save(auten.state_dict(), save_path + "Autoencoder_"+datetime.datetime.now().strftime("%b%d_%H_%M_%S")+".dat")