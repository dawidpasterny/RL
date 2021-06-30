import sys
from contextlib import closing
import itertools as it
from fractions import Fraction
from matplotlib.animation import FuncAnimation
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

from gym.utils import seeding
from gym import Env, spaces, ObservationWrapper

m = np.zeros((24,24))
m[4,0] = 0.587
m[4,-1] = 0.299
m[0:16,11:14]=1

MAPS = {"16x8":{"map":np.array([[0,0,0,0,0,0,0,0],
                                [0.587,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0.299],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0]]),
                "i_target":1, "i_start":1},
        "24x24Wall":{"map":m, "i_target":2, "i_start":1}}
        

def generate_map(n, rng=None):
    """ Generates a world with specified shape and aspect ratio - the aspect ratio
        will not change the shape but rather just add "margins" -
        a number of obstacles with specified size and random location of the input 
        at the left border and output at the right according to the seed.
    """
    # np.random.seed(seed)
    world = np.zeros((n,n))

    # Margins, start & finish
    sy,fy = rng.randint(1,n-1, size=2) # [incl, excl)
    sx,fx = 0,-1

    if rng.choice([0,1]): # if margins
        m = rng.randint(n*0.2, n*0.6)//2 # 20% to 60% taken up by margins
        # print(f"Margin {m}")
        if rng.choice([0,1]): # horizontal or vertical margins
            world[0:m,:]=1
            world[-m::,:]=1
            sy,fy = rng.randint(m+1,n-m-1, size=2) 
        else:
            world[:, 0:m]=1
            world[:, -m::]=1
            sx,fx = m, n-m-1

    world[sy,sx] = 0.587 # green in grayscele
    world[fy,fx] = 0.299 # red in grayscele

    # Obstacles
    def add_obstacle(world, p1, p2):
        n,m = world.shape
        x_idxs = np.arange(n*p1[0], m*p2[0], np.sign(p2[0]-p1[0]), dtype=int)
        y_idxs = np.arange(n*p1[1], m*p2[1], np.sign(p2[1]-p1[1]), dtype=int)
        X,Y = np.meshgrid(y_idxs,x_idxs)
        world[X,Y] = 1

    # # add obstacles till 30% of the world is not accesible
    # while np.count_nonzero(world)/world.size < 0.3:
    #     obst = np.random.random(2) # vertices of obstacles

    # Ratios
    i_target, i_start = rng.choice(RATIOS), rng.choice(RATIOS)
    return world, i_target, i_start


def get_admisible_ratios(steps, n=4):
    """ First calculates admisible ratio for one gear stage given a number of steps
        (up and down) then calculates their n products to account for different combinations
    """
    # Ratios for traditional gear stage
    r = [-Fraction(*t) for t in it.product(np.arange(steps)+1,np.arange(steps)+1)]
    # For planetery gear stage
    for i in range(2,steps+1): # ring radius
        for j in range(1,i): # planete diamter
            r.append(Fraction(2*i-j,i-j))      
    # check for repetitions
    ratios=[]
    for frac in r:
        if frac not in ratios:
            ratios.append(frac)

    # stage_ratios=ratios[::]
    # for _ in range(n):
    #     for frac in [r1*r2 for r1,r2 in it.product(ratios,stage_ratios)]:
    #         if frac not in ratios:
    #             ratios.append(frac)
    return ratios

STEPS = 4 # 6 discrete steps in every direction
N = 24
RATIOS = get_admisible_ratios(STEPS)

class GridSchematic2D(Env):
    def __init__(self, map_name=None, seed=None):
        self.seed(seed)
        if map_name is None:
            self.map, self.i_target, self.i_start = generate_map(n = N, rng=self.np_random)
        else:
            self.map = MAPS[map_name]["map"]
            self.i_target = Fraction(MAPS[map_name]["i_target"]) # target ratio
            self.i_start = Fraction(MAPS[map_name]["i_start"]) # target ratio
        self.p_start = np.nonzero(np.array(self.map==0.587).ravel())[0][0] # initial position
        self.p_target = np.nonzero(np.array(self.map==0.299).ravel())[0][0] # target position
        self.nrow, self.ncol = self.map.shape

        nA = 4*STEPS
        nS = self.nrow * self.ncol


        # State and action space (must be stationary)
        # self.observation_space = spaces.Dict({"position": spaces.Discrete(nS),\
        #                                     "target_ratio": spaces.Discrete(len(ratios)),\
        #                                     "current_ratio": spaces.Discrete(len(ratios))})
        self.observation_space = spaces.Tuple((spaces.Discrete(nS), spaces.Discrete(len(RATIOS)), spaces.Discrete(len(RATIOS))))
        self.action_space = spaces.Discrete(nA)

        fig,ax = plt.subplots(figsize=[8,8])
        ax.set_aspect('equal')
        fig.tight_layout()
        self.fig, self.ax = fig, ax
        self.reset()
        # assert self.i_target in RATIOS
        # assert self.i_start in RATIOS
        return


    def step(self, a):
        """Termination if:
            - it walked into itself
            - it arrived to the finish
            - if ring gear radius = planet gear diamater
            No termination (but ultimately reduction in reward) if:
            - it tried to walk into an obstacle or map boundary
            - gear stage doesn't fit into the space,
            - agent moves horizontally after creating a single gear (i.e single step up).

            Returns tuple: (new_state, reward, done)
        """
        i = self.s[2] # current ratio
        a_last =self.last_action
        p = self.s[0] # current position
        reward = 0
        done = False
        row, col = p//self.ncol, p%self.ncol
        step = a//4+1 # distance to travel (in some direction)
        last_step = a_last//4+1
        info = None
        try:
            if a%4==0: # going left
                # It is possible to create a shaft after just one gear has been placed
                # it doesn't consitute a gear stage and diminishes reward but it's possible
                new_col=col-step
                if self.no_go[row,new_col]==1:
                    done = True
                    info = "Termination, walked into a itself"
                    reward = -1 + col/self.ncol
                elif new_col>=0 and self.map[row,new_col]!=1: 
                    # if we don't go into a wall or obstacle
                    p -= step
                    self.no_go[row, new_col+1:col+1] = 1
                else: # in case of silent fail, ignore the current action
                    a=a_last
            elif a%4==2: # going right
                new_col = col+step
                if self.no_go[row,new_col]==1:
                    done = True
                    info = "Termination, walked into itself"
                    reward = -1 + col/self.ncol
                elif new_col<self.ncol and self.map[row,new_col]!=1:
                    p += step
                    self.no_go[row,col:new_col] = 1
                else: # in case of silent fail, ignore the current action
                    a=a_last
            else: # going up or down
                if a_last%4==0 or a_last%4==2: # no gear created in previous move
                    # check if a single gear fits
                    if any(self.no_go[row-step:row+step+1,col].ravel()==1):
                        done = True
                        info = "Termination, walked into itself"
                        reward = -1 + col/self.ncol
                    elif row >= step and row+step < self.nrow and all(self.map[row-step:row+step+1, col].ravel()!=1):
                        self.no_go[row-step:row+step+1,col] = 1
                    else: # in case of silent fail, ignore the current action
                        a=a_last
                else: # creating a gear stage
                    # will transition to new row only if a valid gear stage has been created
                    # i.e. if previus step was also a vertical one
                    if a%4==1 and a_last%4==1: # up-up, traditional gear stage
                        new_row = row-last_step-step-1
                        if any(self.no_go[new_row-step:new_row+step+1,col].ravel()==1):
                            done = True
                            info = "Termination, walked into itself"
                            reward = -1 + col/self.ncol
                        elif new_row-step >= 0 and all(self.map[new_row-step:new_row+step+1, col].ravel()!=1): 
                            # advance the position if there is enough space
                            p -= (last_step+step+1)*self.ncol
                            i *= -Fraction(last_step,step)
                            self.no_go[new_row-step:new_row+step+1,col] = 1
                        else: # in case of silent fail, ignore the current action
                            a=a_last
                    elif a%4==3 and a_last%4==3: # down-down, traditional gear stage
                        new_row = row+last_step+step+1
                        if any(self.no_go[new_row-step:new_row+step+1,col].ravel()==1):
                            done = True
                            info = "Termination, walked into itself"
                            reward = -1 + col/self.ncol
                        elif new_row+step < self.nrow and all(self.map[new_row-step:new_row+step+1, col].ravel()!=1): 
                            # advance the position if there is enough space
                            p += (last_step+step+1)*self.ncol
                            i *= -Fraction(last_step,step)
                            self.no_go[new_row-step:new_row+step+1,col] = 1
                        else: # in case of silent fail, ignore the current action
                            a=a_last
                    else: # down-up or up-down, planetary gear stage (thickness=2)
                        if self.no_go[row,col+1]==1: # we came from the right
                            if any(self.no_go[row-last_step-1:row+last_step+2,col-1].ravel()==1) or self.no_go[row-last_step-1,col]==1 or self.no_go[row+last_step+1,col]==1:
                                done = True
                                info = "Termination, walked into itself"
                                reward = -1 + col/self.ncol
                            elif row > last_step and row+last_step+1 < self.nrow and all(self.map[row-last_step-1:row+last_step+2,col-1:col+1].ravel()!=1): 
                                p-=1
                                self.no_go[row-last_step-1:row+last_step+2,col-1:col+1] = 1
                                try:
                                    i *= Fraction(2*last_step-step,last_step-step)
                                except ZeroDivisionError:
                                    done=True
                                    info = "Termination, the planet gear diameter must be smaller than ring gear radius"
                                    reward = -1 + col/self.ncol
                            else: # in case of silent fail, ignore the current action
                                a=a_last
                        else: # came from the left
                            if any(self.no_go[row-last_step-1:row+last_step+2,col+1].ravel()==1) or self.no_go[row-last_step-1,col]==1 or self.no_go[row+last_step+1,col]==1:
                                done = True
                                info = "Termination, walked into itself"
                                reward = -1 + col/self.ncol
                            elif row > last_step and row+last_step+1 < self.nrow and all(self.map[row-last_step-1:row+last_step+2,col:col+2].ravel()!=1): 
                                p+=1
                                self.no_go[row-last_step-1:row+last_step+2,col:col+2] = 1
                                try:
                                    i *= Fraction(2*last_step-step,last_step-step)
                                except ZeroDivisionError:
                                    done=True
                                    info = "Termination, the planet gear diameter must be smaller than ring gear radius"
                                    reward = -1 + col/self.ncol
                            else: # in case of silent fail, ignore the current action
                                a=a_last
        except IndexError:
            done=True
            info="Out of bounds"
                
        # if p%self.ncol == self.ncol-1:
        if p == self.p_target: # solved
            reward = np.exp(-float(self.i_target-i)**2/4) # 1 if achieved target ratio
            # row_target = self.p_target//self.ncol
            # reward = np.exp(-float(p//self.ncol-row_target)**2/4) # 1 if achieved target ratio
            # reward = 1 # just get to the end
            done = True
            info = f"Episode solved with ratio {i}, reward= {reward}!"

        new_state = (p, self.i_target, i)
        self.last_action = a
        self.s = new_state
        return new_state, reward, done, info


    def render(self, mode='human', delay=0.1, record=False):
        plot = self.no_go + self.map
        p = self.s[0]
        plot[p//self.ncol, p%self.ncol]= 0.587
        self.ax.clear()
        self.ax.pcolormesh(np.flip(plot,0), cmap="binary", norm=plt.Normalize(0,1),\
                        edgecolors='grey', linewidths=.1)
        plt.pause(delay)


    def reset(self, random=False):
        if random: # generate new random map
            self.map, self.i_target, self.i_start = generate_map(n=N, rng=self.np_random)
            self.p_start = np.nonzero(np.array(self.map==0.587).ravel())[0][0]
            self.p_target = np.nonzero(np.array(self.map==0.299).ravel())[0][0]
            
        self.s = (self.p_start, self.i_target, self.i_start)
        self.last_action = 2 # a bit of a hack to avoid complication is step()
        self.no_go = np.zeros_like(self.map) # a map of visited states and obstackles
        self.ax.clear()
        return self.s


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print(seed)
        return seed


class ScreenOutput(ObservationWrapper):
    """ Wrapps the state of the environment to a form suitable for CNN,
        returns a tuple of single channel, gray-scale NxN pixel picture in a format 
        required by Pytorch and the relative difference between current and target ratio
    """
    def __init__(self, env=None):
        super().__init__(env)
        # self.observation_space =  spaces.Tuple((spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=np.float32), self.observation_space[-1]))
        self.observation_space =  spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=np.float32)

    def observation(self, obs):
        p = obs[0] # current position
        i = obs[-1] # current ratio
        f = self.p_target
        o = self.no_go+self.map
        # i_diff = float((obs[-1] - self.i_target)/self.i_target)
        o[p//self.ncol, p%self.ncol] = float(i)
        o[f//self.ncol, f%self.ncol] = float(self.i_target)
        # return np.array([o]), i_diff
        return np.array([o], dtype=np.float32)


if __name__=="__main__":
    # env = GridSchematic2D(map_name="24x24Wall")
    env = GridSchematic2D(seed=3672871121734420758)
    env = ScreenOutput(env)
    env.render()
    print(env.seed)
    print(f"Target ratio:{env.i_target}")
    print(f"Current ratio:{env.s[2]}")

    # play manually
    flag = True
    while flag:
        # wsad + distance e.g. 'w3' to go 3 steps up
        f = input("Action or q to quit:")
        try:
            dist = int(f[1])-1
        except:
            dist = 0

        if f[0]=='a':
            action = 4*dist
        elif f[0]=='w':
            action = 1+4*dist
        elif f[0]=='d':
            action = 2+4*dist
        elif f[0]=='s':
            action = 3+4*dist
        elif f[0]=='q':
            flag=False

        new_state, reward, done, info = env.step(action)
        print(info if info!=None else '')
        if done:
            env.reset(random=True)
            print("Starting again with new env...\n")
        print(f"Target ratio:{env.i_target}")
        print(f"Current ratio:{env.s[2]}")
        env.render()

    # # Play random episodes
    # env = GridSchematic2D()
    # print(f"Current ratio:{env.s[2]}")
    # env.render()
    # e = 10 # 10 episodes
    # rewards=[]
    # while e!=0:
    #     a = env.action_space.sample()
    #     new_state, reward, done, _ = env.step(a)
    #     if done:
    #         rewards.append(reward)
    #         env.reset(random=True)
    #         print(f"Current ratio:{env.s[2]}")
    #         env.render()
    #         e -= 1
    #     env.render(delay=1)
