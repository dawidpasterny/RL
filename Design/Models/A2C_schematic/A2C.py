import sys
import os
sys.path.append(os.getcwd())
from Design.Environments import y_schematic_reward as ys

from collections import namedtuple, deque, defaultdict
import numpy as np
import datetime
import itertools as it

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 8 # number of episodes to play before swtiching to training
UNROLL_STEPS = 4
ENTROPY_BETA = 1e-3
ENTROPY_BETA_STOP = 5e5
CLIP_GRAD = 10

PATH = "./Design/Models/A2C_schematic/"


class A2CNet(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        b = self.net(x)
        a = self.policy_head(b)
        v = self.value_head(b)
        return a,v


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done'))
ExperienceDiscounted = namedtuple('ExperienceDiscounted', ('state', 'action', 'total_reward', 'last_state'))

class ExperienceGenerator():
    """ Generates n-step, discounted experiences by acting in a single or multiple environments
        (in the latter case we swich between them in a round robin fashion).
        Returns ExperienceDiscounted tuple.
    """
    def __init__(self, env, agent, gamma, unroll_steps):
        self.agent = agent
        self.gamma = gamma # discount factor
        self.unroll_steps = unroll_steps
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        # Running average reward and steps
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = deque(maxlen=100)


    def __iter__(self):
        states, trajectories, cur_rewards, cur_steps = [], [], [], []
        for env in self.pool:
            states.append(env.reset())
            trajectories.append(deque(maxlen=self.unroll_steps))
            cur_rewards.append(0.0)
            cur_steps.append(0)

        while True:
            actions = self.agent(states) # return a list of actions
            for env_idx, (env, action) in enumerate(zip(self.pool, actions)):
                next_state, reward, is_done = env.step(action)
                trajectory = trajectories[env_idx]
                cur_rewards[env_idx] += reward
                cur_steps[env_idx] += 1
                trajectory.append(Experience(state=states[env_idx], action=action, reward=reward, done=is_done))

                if len(trajectory) == self.unroll_steps:
                    yield self.get_discounted_experience(list(trajectory), self.gamma)
                states[env_idx] = next_state
                if is_done:
                    # in case of very short episode (shorter than our steps count), send gathered history
                    if len(trajectory) < self.unroll_steps:
                        yield self.get_discounted_experience(trajectory, self.gamma)
                    while len(trajectory) > 1: # exhaust current trajectory
                        trajectory.popleft()
                        yield self.get_discounted_experience(trajectory, self.gamma)
                    self.episode_rewards.append(cur_rewards[env_idx])
                    self.episode_steps.append(cur_steps[env_idx])
                    cur_rewards[env_idx] = 0.0
                    cur_steps[env_idx] = 0
                    states[env_idx] = env.reset()
                    trajectory.clear()

    @staticmethod
    def get_discounted_experience(traj, gamma):
        if traj[-1].done:
            last_state = None
        else:
            last_state = traj[-1].state
            # we don't wan't to include reward of the last step of the trajectory if it's not
            # terminated because we gonna approximate the value of that state instead later
            traj = traj[:-1]
        discounted_reward = 0.0
        for exp in reversed(traj):
            discounted_reward *= gamma
            discounted_reward += exp.reward
        return ExperienceDiscounted(traj[0].state, traj[0].action, discounted_reward, last_state)


    def get_steps_and_rewards(self):
        if len(self.episode_rewards)>0:
            mean_reward = np.mean(self.episode_rewards)
            mean_steps = np.mean(self.episode_steps)
        else:
            mean_reward, mean_steps = 0,0
        return mean_reward, mean_steps


class PolicyAgent():
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def __call__(self, states):
        """ Passes a batch of states through a model to get a probability
            distribution over actions from which, for every state a specific
            action is sampled.
        """
        split_idxs = [len(ys.DIAMS), len(ys.DIAMS), len(ys.PHI)]
        states = torch.FloatTensor(states).to(self.device) # preprocessing of data for NN
        scores = self.model(states) # returns a batch of raw scores, no softmax applied
        # print(scores)
        actions = []
        for score in scores:
            a = []
            for s in torch.split(score, split_idxs):
                prob = nn.functional.softmax(s-max(s), dim=0).numpy()
                a.append(np.random.choice(len(prob), p=prob))
            actions.append(a)
        return actions


def unpack_batch(batch, net, device='cpu'):
    """ Converts a batch of experiences into training batch, takes
        care of value approximation
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        # reward is the already accumulated, total, discounted reward on trajectory of length REWARD_STEPS
        rewards.append(exp.total_reward) 
        if exp.last_state is not None: # None if termination occurs within unrolled trajectory
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_t = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idx: # if there are any experiences that didn't terminate upon unrolling
        last_states = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals = net(last_states)[1]
        last_vals_np = last_vals.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** UNROLL_STEPS # Approximate the value of the last state
        rewards_np[not_done_idx] += last_vals_np # Q = G_t + V(s_last)

    vals_t = torch.FloatTensor(rewards_np).to(device)

    return states_t, actions_t, vals_t


class Tracker():
    def __init__(self, writer, batch_size):
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, name, value, index):
        assert isinstance(name, str)
        assert isinstance(index, int)

        data = self._batches[name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(name, np.mean(data), index)
            data.clear()


if __name__ == "__main__":
    env = ys.GridSchematic3D(seed=0, static=False)
    # run tensorboard --logdir=./Design/Models/A2C/runs
    writer = SummaryWriter(log_dir=PATH+"runs/A2C2D"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))
    net = A2CNet(env.observation_space.shape[0], sum([a.n for a in env.action_space]))
    print(net)
    device = "cpu"

    agent = PolicyAgent(lambda x: net(x)[0])
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    exp_gen = ExperienceGenerator(env, agent, gamma=GAMMA, unroll_steps=UNROLL_STEPS)

    split_idxs = [len(ys.DIAMS), len(ys.DIAMS), len(ys.PHI)]
    best_mean_reward = None
    exp_batch = []


    with Tracker(writer,10) as tracker:
        for step_idx, exp in enumerate(exp_gen):
            exp_batch.append(exp)
            # print(exp)

            if len(exp_batch) < BATCH_SIZE:
                continue

            mean_episode_reward, mean_step_count = exp_gen.get_steps_and_rewards()
            writer.add_scalar("episode_reward_100", mean_episode_reward, step_idx)
            writer.add_scalar("episode_step_100", mean_step_count, step_idx)

            if mean_episode_reward > 0.95:
                print(f"Solved after {step_idx} steps!")
                break

            if best_mean_reward is None or best_mean_reward < mean_episode_reward:
                torch.save(net.state_dict(), PATH+"A2C2D-best.dat")
                print(f"Best mean reward updated -> {mean_episode_reward:.3f}, model saved")
                best_mean_reward = mean_episode_reward

            states_t, actions_t, vals_t = unpack_batch(exp_batch, net, device=device)
            # print(exp_batch)
            # print(states_t)
            # print(actions_t)
            # print(vals_t)
            exp_batch.clear()

            optimizer.zero_grad()
            logits, values = net(states_t)

            # Value loss
            value_loss = nn.functional.mse_loss(values.squeeze(-1), vals_t)
            adv = vals_t - values.detach() # advantages   

            # Policy and entropy loss
            entropy_beta = max(0, ENTROPY_BETA*(1-step_idx/ENTROPY_BETA_STOP))
            entropies = torch.zeros(len(states_t))
            log_prob = torch.zeros(len(states_t))
            for i,l in enumerate(torch.split(logits, split_idxs, dim=1)):
                log_distr = nn.functional.log_softmax(l, dim=1)
                distr = nn.functional.softmax(l-torch.max(l, dim=0)[0], dim=1) # -max for better stability
                # Joint entropy (of indepentend random variables)
                entropies += torch.sum(distr*log_distr, dim=1)
                # Log of joint probabilities of independent random variables
                log_prob += log_distr[range(len(l)), actions_t[range(len(l)),i]]
            log_prob_actions = adv * log_prob
            policy_loss = -log_prob_actions.mean()
            entropy_loss = entropy_beta * entropies.mean()

            # Policy gradients magnitude for inspection
            policy_loss.backward(retain_graph=True) # retain graph for gradient descent
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in net.parameters()
                                    if p.grad is not None])


            # Entropy and value gradients
            loss = entropy_loss + value_loss

            # Training
            loss.backward() # updates weights on
            nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD) # to awoid blow-up
            optimizer.step()
            loss += policy_loss # total loss

            # Tracking
            tracker.track("advantage", adv, step_idx)
            tracker.track("values", values, step_idx)
            tracker.track("entropy_loss", entropy_loss, step_idx)
            tracker.track("policy_loss", policy_loss, step_idx)
            tracker.track("value_loss", value_loss, step_idx)
            tracker.track("loss_total", loss, step_idx)
            tracker.track("PG_L2", np.sqrt(np.mean(np.square(grads))), step_idx)
            tracker.track("PG_max", np.max(np.abs(grads)), step_idx)
            tracker.track("PG_var", np.var(grads), step_idx)
