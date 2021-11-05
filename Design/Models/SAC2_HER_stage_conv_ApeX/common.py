import sys
import time
import numpy as np
np.set_printoptions(precision=4, threshold=sys.maxsize)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import Design.Environments.discrete_stage_creator_unwrapped as sc

from collections import namedtuple, deque, defaultdict

# if next_state=None means s was a terminal state
Experience = namedtuple('Experience', ['screen', 'state', 'action', 'reward', 'done', 'next_screen', 'next_state'])

class Agent_SAC_HER():
    def __init__(self, act_net, env, buffer, gamma, device="cpu", **kwargs):
        self.act_net = act_net
        self.env = env # actor network
        self.exp_buffer = buffer
        self.device = device
        # Running average reward and steps
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = deque(maxlen=100)
        #Misc
        # clip d to 0.02 but env terminates if d<0.05, that way hopefully it will 
        # learn not to take crazy small d
        # self.clip = lambda x: [min(max(0.02,x[0]), .9), min(max(0,x[1]), 1)] 
        self.clip = lambda x: [sc.DIAMS[min(max(2,round(x[0]/0.05)),5)-2], sc.ANGLES[min(max(0,round(x[1]/(np.pi/3))),5)]] # casts to admisible action
        self.unroll_steps = kwargs.get("unroll_steps", 1)
        self.gamma = gamma # for discounting
        # self.fig, self.ax = plt.subplots(1,1)


    @torch.no_grad()
    def play_episode(self, random=True):
        """ Play an entire episode (because of HER and since they won't get much longer either).
            Appends the real and manufactured trajectories to the experience buffer.

            Returns the number of newly added experiences (i.e. steps)
        """
        screen, state = self.env.reset(random=random, target=False)["observation"]
        self.env._set_reachable_target()
        state = np.hstack((state, self.env.goal))
        
        total_reward = 0.0
        done = False
        steps=0
        local_buffer=[]

        while not done:
            # print("State: ",state)
            state_t = torch.FloatTensor(np.array([state])).to(self.device)
            screen_t = torch.FloatTensor(np.array([screen])).to(self.device)

            mu, std = self.act_net(screen_t, state_t) 
            action = np.random.normal(mu, std)[0]
            action = self.clip(action) # [d, phi]
            
            # Perform step
            next_obs, reward, done, info = self.env.step(action)
            if info.get("invalid", False):
                continue # not to append same states over and over again

            next_state = np.hstack((next_obs["observation"][1], self.env.goal))
            next_screen = next_obs["observation"][0]
            total_reward += reward
            # print("Next state: ", next_state)

            local_buffer.append([screen, state, action, reward, done, next_screen, next_state])
            # self.exp_buffer.append(Experience(*local_buffer[-1]))
            # print("Exp: ", np.array(local_buffer[-1], dtype=object)[[1,2,3,4,6]])
            state = next_state
            screen = next_screen
            steps+=1
            
        # Unroll from only the last state since other rewards are 0 either way
        # exp = [screen, state, action, reward, done, next_screen, next_state]
        if steps>self.unroll_steps:
            for i in range(steps-self.unroll_steps):
                # reward is 0 so I don't care about discounting
                local_buffer[i][-3:] = local_buffer[i+self.unroll_steps-1][-3:]     
                # print("Exp: ", np.array(local_buffer[i], dtype=object)[[1,2,3,4,6]])
                self.exp_buffer.append(Experience(*local_buffer[i]))
        for i in range(min(steps, self.unroll_steps),1,-1): # Reward discouting here
            local_buffer[-i][-3:] = local_buffer[-1][-3:]
            local_buffer[-i][3] = reward*self.gamma**(i-1)
            # print("Exp: ", np.array(local_buffer[-i], dtype=object)[[1,2,3,4,6]])
            self.exp_buffer.append(Experience(*local_buffer[-i]))


        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)

        
        # HER, substitute target with the second to last state
        # Using unrolling and HER might actually be the only wayt to overcome the limbo of first 
        # step where any action won't change the state because one gear doesn't make up a gear stage
        n = len(local_buffer)
        if n>2: # >2 because the first action doesn't change the state
            # .pop(), do not to take the experience that terminated
            last_screen, last_state, _,_,_,_,_ = local_buffer.pop() # new terminals
            target = last_state[:3] # x_current, y_current, i_current
            # First, fix the unrolling after deleting the last experience
            # but do so only for the last unroll_steps steps because these are the
            # experiences where HER actually matters, there is no points appending new experiences 
            # with new target but reward still 0
            for i in range(self.unroll_steps):
                local_buffer[-1-i][-3:] = [True, last_screen, last_state]
                local_buffer[-1-i][3] = self.gamma**(i) # reward in HER is 1
                # Then fix the targets of state and next_state and append to the buffer
                # state = (x_current, y_current, i_current, x_target, y_target, i_target)
                local_buffer[-1-i][1][-3:] = target # state target
                local_buffer[-1-i][-1][-3:] = target # next_state target
                self.exp_buffer.append(Experience(*local_buffer[-1-i]))
                steps +=1

        # if done:
        #     print("Done\n")

        return steps


    def get_mean_reward_and_steps(self):
        if len(self.episode_rewards)>0:
            mean_reward = np.mean(self.episode_rewards)
            mean_steps = np.mean(self.episode_steps)
        else:
            mean_reward, mean_steps = 0,0
        return mean_reward, mean_steps


class ExperienceBuffer:
    def __init__(self, buffer_size, device="cpu"):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def pop(self):
        self.buffer.pop()

    def sample(self, batch_size):
        """ Samples a batch of experiences and unpacks them  """
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        screens ,states, actions, rewards, dones, next_screens, next_states \
                                    = zip(*[self.buffer[idx] for idx in indices])

        screens_t = torch.tensor(np.array(screens, dtype=np.float32)).to(self.device)
        states_t = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        actions_t = torch.tensor(np.array(actions, dtype=np.int32)).to(self.device)
        rewards_t = torch.tensor(np.array(rewards, dtype=np.float32)).to(self.device)
        next_screens_t = torch.tensor(np.array(next_screens, dtype=np.float32)).to(self.device)
        next_states_t = torch.tensor(np.array(next_states, dtype=np.float32)).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        # print(torch.count_nonzero(dones_t))

        return screens_t, states_t, actions_t, rewards_t, dones_t, next_screens_t, next_states_t



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
