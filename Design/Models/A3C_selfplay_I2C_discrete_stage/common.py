from collections import deque, defaultdict

import torch
from torch._C import device
import torch.nn.functional as F
import numpy as np

import Design.Environments.discrete_stage_creator_unwrapped as sc


class SelfplayAgent():
    """ On-policy DDPG agent using selfplay and Orstein-Uhlenbeck process for meta
        exploration of the taskgiver.
    """
    def __init__(self, alice, bob, env, device="cpu", **kwargs):
        # Both alice and bob are architectually almost same except that alice doesn't 
        # get target states, she is the one creating them for bob in fact
        self.alice = alice # task giver
        self.bob = bob # worker
        self.device = device
        # self.unroll_steps = kwargs.get("unroll_steps", 1)
        self.gamma = kwargs.get("gamma", .99) # discounting
        self.env = env
        # Running average reward and steps
        self.a_rewards = deque(maxlen=100)
        self.b_rewards = deque(maxlen=100)
        # For target task
        # self.episode_rewards = deque(maxlen=100)
        # self.episode_steps = deque(maxlen=100)
        # clip d to 0.02 but env terminates if d<0.05, that way hopefully it will 
        # learn not to take crazy small d
        self.clip = lambda x: [min(max(0.05,x[0]), .7), min(max(0,x[1]), 1)] 
        # self.clip = lambda x: [min(max(0.05,x[0]), .8), x[1]%1] 


    def selfplay_episode(self, random=True):
        """ Alice and Bob play against each other. Both play entire episodes at a
            time, their trajectories tracked, converted into tensors, unrolled and
            returned as one entry.
            - random: bool - wheather to keep changing the map or keep it constant
            It's more of a REINFORCE with Q baseline than AC
            No multiple environments since env is randomly generated each time
            
        """
        a_screens, a_states, a_actions, a_rewards = [], [], [], []
        b_screens, b_states, b_actions, b_rewards = [], [], [], []
        a_steps, b_steps = 0, 0 # used to calculated the reward
        done = False
        stop = False
        # p_target and i_target are initialized to p_start and 1.0
        screen, state = self.env.reset(target=False, random=random)["observation"]
        state = np.hstack((state, self.env.goal))
        # self.env.render()

        # Everything here happens on cpu
        # print("Alice's turn")
        while not done: # if MAX_STEPS reached
            a_steps += 1
            a_states.append(state)
            a_screens.append(screen)
            state_t = torch.FloatTensor([state]).to(self.device)
            screen_t = torch.FloatTensor([screen]).to(self.device)

            # Act
            logits, _, stop_logits = self.alice(screen_t, state_t)
            # probs = F.softmax(logits-max(logits), dim=1)[0].detach().numpy()
            probs = F.softmax(logits, dim=1)[0].cpu().detach().numpy()
            # stop_probs = F.softmax(stop_scores-max(stop_scores), dim=1)[0].detach().numpy()
            stop_probs = F.softmax(stop_logits, dim=1)[0].cpu().detach().numpy()
            act_idx = np.random.choice(len(probs), p=probs)
            stop = np.random.choice(len(stop_probs), p=stop_probs)
            action = [act_idx, stop]

            if stop:
                if len(self.env.traj)>1:
                # Stopping at a right moment is what Alice needs to learn actually
                # print(f"Alice stopped after {len(self.env.traj)} gears, {a_steps} steps")
                    break
                else:
                    # pretend no stop has been played if the trajectory is too short
                    action[-1]=0 

            # Perform step +[1] to indicate that it's alice
            obs, _, done, info = self.env.step(action+[1])
            a_actions.append(action)
            screen, state = obs["observation"]
            state = np.hstack((state, self.env.goal))
            # print(state)
            # self.env.render()
        
        # print("Bob's turn")
        done=False
        self.env.set_target(state[:2], state[2])
        screen, state = self.env.reset(random=False)["observation"] # keep the same env_map
        state = np.hstack((state, self.env.goal))
        # self.env.render()


        while not (done or a_steps+b_steps>=sc.MAX_STEPS):
            b_steps += 1
            b_states.append(state)
            b_screens.append(screen)
            state_t = torch.FloatTensor([state]).to(self.device)
            screen_t = torch.FloatTensor([screen]).to(self.device)

            # Act
            logits, _, _ = self.bob(screen_t, state_t)
            # probs = F.softmax(logits-max(logits), dim=1)[0].detach().numpy()
            probs = F.softmax(logits, dim=1)[0].cpu().detach().numpy()
            action = [np.random.choice(len(probs), p=probs)]
            b_actions.append(action)

            # Perform step +[0] to indicate that it's bob
            obs, reward, done, _ = self.env.step(action+[0])
            screen, state = obs["observation"]
            state = np.hstack((state, self.env.goal))
            # self.env.render()
            # if done and reward==0: # bob could have just occluded the output, not solve the env
            #     b_steps = sc.MAX_STEPS-a_steps
            #     # print(f"Bob solved an environment with {len(self.env.traj)} gears")

        # Calculate rewards (Monte Carlo). Reward is sparse, given only at the end of the episode
        r_b = -b_steps
        for i in range(b_steps-1,-1,-1): # (b_steps,0]
            b_rewards.append(r_b*self.gamma**i)

        r_a = max(0, b_steps-a_steps)
        for i in range(a_steps-1,-1,-1):
            a_rewards.append(r_a*self.gamma**i)

        self.a_rewards.append(r_a)
        self.b_rewards.append(r_b)

        # Convert to tensors and move to GPU
        a_data = [torch.Tensor(x) for x in [a_screens, a_states, a_actions, a_rewards]]
        b_data = [torch.Tensor(x) for x in [b_screens, b_states, b_actions, b_rewards]]
        return  a_data, b_data
        

    def test_bob(self, num_tests):
        """ Test Bob on target environments (not generated by Alice and w/o exploration)"""

        rewards = 0.0
        steps = 0
        for _ in range(num_tests):
            screen, state = self.env.reset()
            # print("Init state: ", state)
            while True: # play a full episode
                state_t = torch.tensor([state])
                screen_t = torch.tensor([screen])
                # Just take the mean (no exploration)
                action = self.bob(screen_t, state_t)[0].data.numpy()
                # print("Action: ", action)
                (screen, state), reward, done, _ = self.env.step(action)
                # print("Next state: ", state)
                steps += 1
                if done:
                    rewards += reward
                    break
        return rewards/num_tests, steps/num_tests



    def get_selfplay_rewards(self):
        mean_r_a= np.mean(self.a_rewards)
        mean_r_b = np.mean(self.b_rewards)
        return mean_r_a, mean_r_b


class Tracker():
    def __init__(self, writer, batch_size):
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = defaultdict(list)
        return self

    def __exit__(self):
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