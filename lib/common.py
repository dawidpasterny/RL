import sys
import time
import numpy as np
np.set_printoptions(precision=4, threshold=sys.maxsize)

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import namedtuple, deque, defaultdict
# import matplotlib.pyplot as plt

# if next_state=None means s was a terminal state
Experience = namedtuple('Experience', ['screen', 'state', 'action', 'reward', 'done', 'next_screen', 'next_state'])

class AgentDDPG():
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, act_net, env, buffer, ae, gamma, device="cpu", **kwargs):
        self.act_net = act_net
        self.env = env # actor network
        self.exp_buffer = buffer
        self.device = device
        self.ae = ae
        # Running average reward and steps
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = deque(maxlen=100)
        # Orsetein-Uhlenbeck process parameters
        self.ou_mu = kwargs.get("ou_mu", 0.0)
        self.ou_teta = kwargs.get("ou_teta", 0.15)
        self.ou_sigma = kwargs.get("ou_sigma", np.array([0.2, 0.4])) # variance, originally .2
        self.ou_epsilon = kwargs.get("ou_epsilon", .25) # originally 1.0
        self.a_state = np.zeros(env.action_space.shape) # agent state for OU
        #Misc
        # clip d to 0.02 but env terminates if d<0.05, that way hopefully it will 
        # learn not to take crazy small d
        # self.clip = lambda x: [min(max(0.02,x[0]), 1), min(max(0,x[1]), 1)] 
        self.clip = lambda x: [min(max(0.02,x[0]), 1), x[1]%1] 
        self.unroll_steps = kwargs.get("unroll_steps", 1)
        self.gamma = gamma # for unrolling
        # self.fig, self.ax = plt.subplots(1,1)


    def play_episode(self):
        """ Play an entire episode using OU exploration, append the casual and HER 
            experiences to the buffer. (using her is the reason we need to play entire episodes)
            
            reduce dimensionality before storage? problem the features would be old?

            Returns the number of newly buffered experiences
        """
        screen,state = self.env.reset()
        total_reward = 0.0
        done = False
        steps=0
        local_buffer=[]
        self.a_state = np.zeros(self.env.action_space.shape)

        while not done:
            # print("State: ",state)
            state_t = torch.tensor([state]).to(self.device).float()
            screen_t = torch.tensor([screen]).to(self.device)
            features = torch.reshape(self.ae(screen_t), (1,-1)).float()
            action_t = self.act_net(torch.column_stack((features, state_t))) # actions tensor
            action = action_t[0].data.cpu().numpy()
            # action1=action.copy()

            # Orstein-Uhlenbeck step
            if self.ou_epsilon > 0:
                self.a_state += self.ou_teta * (self.ou_mu - self.a_state) \
                        + self.ou_sigma * np.random.normal(size=action.shape) # random noise

                action += self.ou_epsilon * self.a_state
                action = self.clip(action)
            # No random process, just normal noise
            # action += self.ou_sigma * np.random.normal(.5, .1, size=action.shape)
            # action = self.clip(action)
            
            # print(f"{action1}, \t {self.ou_epsilon*self.a_state}, \t {action}")

            # Perform step
            (next_screen, next_state), reward, done, _ = self.env.step(action)
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
        for i in range(min(steps, self.unroll_steps),0,-1): # Reward discouting here
            local_buffer[-i][-3:] = local_buffer[-1][-3:]
            local_buffer[-i][3] = reward*self.gamma**(i)
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
            for i in range(min(self.unroll_steps, len(local_buffer))):
                local_buffer[-1-i][-3:] = [True, last_screen, last_state]
                local_buffer[-1-i][3] = self.gamma**(i) # reward in HER is 1
            # Then fix the targets of state and next_state in every experience
            # state = (x_current, y_current, i_current, x_target, y_target, i_target)
            for i, exp in enumerate(local_buffer):
                exp[1][-3:] = target # state target
                exp[-1][-3:] = target # next state target
                # print("HER experienece:", np.array(exp, dtype=object)[[1,2,3,4,6]])
                self.exp_buffer.append(Experience(*exp))
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


    #         traj.append([screen, state, action, reward, done, next_screen, next_state])
    #         if len(self.traj)==self.unroll_steps:
    #             local_buffer.append(self.get_discounted_experience(traj, self.gamma))
    #         state = next_state
    #         screen = next_screen
    #         steps+=1
    #         if done:
    #             # in case of very short episode (shorter than our steps count), send gathered history
    #             if len(trajectory) < self.unroll_steps:
    #                 local_buffer.append(self.get_discounted_experience(traj, self.gamma))
    #             while len(trajectory) > 1: # exhaust current trajectory
    #                 trajectory.popleft()
    #                 local_buffer.append(self.get_discounted_experience(traj, self.gamma))
            
    #     self.episode_rewards.append(total_reward)
    #     self.episode_steps.append(steps)

    #     # HER, substitute target with the second to last state
    #     n = len(local_buffer)
    #     if n!=1:
    #         # .pop() not to take the experience that terminated
    #         target = local_buffer.pop()[1][-3:] # x_target, y_target, i_target
    #         for i, exp in enumerate(local_buffer):
    #             exp[1][-3:] = target # state target
    #             exp[-1][-3:] = target # next state target
    #             # Use unrolling for HER too (hard coded because reward is given 
    #             # only at the very last step
    #             exp[3] = self.gamma**(n-2-i) if n-2-i<self.unroll_steps else 0 # reward
    #             self.exp_buffer.append(Experience(*exp))
    #             steps +=1

    #     return steps


    # @staticmethod
    # def get_discounted_experience(traj:list, gamma):
    #     """ Adds the discounted experience to the self.exp_buffer (of type Experience)
    #         and returns raw experience to be used in HER.
    #         Utilizes the fact that rewards are sparse simplifiying discounting greatly.
    #         exp = [screen, state, action, reward, done, next_screen, next_state]
    #     """
    #     if traj[-1][4]: # done
    #         traj[0][3] = traj[-1][3]*self.gamma**(len(traj)-1) # discounted reward
    #     # if not done the reward is 0 either way so we simply concatenate parts of 
    #     # first and last experience of the traj accordingly
    #     exp = traj[0][:4] + traj[-1][-3:]
    #     self.exp_buffer.append(Experience(*exp))
    #     return exp


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
        actions_t = torch.tensor(np.array(actions, dtype=np.float32)).to(self.device)
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
