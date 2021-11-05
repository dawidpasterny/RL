import sys
import os
sys.path.append(os.getcwd())

import matplotlib
# matplotlib.use('TkAgg') # shows in the browserpip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from fractions import Fraction
import itertools as it
# from scipy.spatial import Delaunay, ConvexHull
import torch

from gym import GoalEnv, spaces, ObservationWrapper, Wrapper
from gym.utils import seeding


def get_admisible_ratios(d):
    """ First calculates admisible ratio for one gear stage given a number of steps
        (up and down) then calculates their n products to account for different combinations
    """
    # Ratios for traditional gear stage
    r = [Fraction(*t) for t in it.product(np.arange(d)+1,np.arange(d)+1)]
     
    # check for repetitions
    ratios=[]
    for frac in r:
        if frac not in ratios:
            ratios.append(float(frac))

    return ratios


def generate_random_map(rng, w, h, boundary=0.2, target=True):
    """ A map is a dictionary of boundary points, start and target (2D) coordinates 
        and target ratio. The csys for a map is given in the lower left corner so 
        that all coordinates are positive
        - rng: random number generator
        - w,h: width and height of the map
        - boundary: probability of creating a map with boundary
        - output: whether a map should contain target position and ratio or not
    """
    m ={}
    # Boundary polygon, parhaps not the most efficient way to do so
    if rng.rand()<boundary:
        # if rng.rand()<.5: # polygon
        #     # Generate a number of random points in [0,1]*w x [0,1]*h
        #     points = rng.rand(25,2)*[w,h]
        #     poly = ConvexHull(points).vertices
        #     # Chose start and target points from the points within the hull
        #     m["p_start"] = points[rng.choice(np.delete(np.arange(25), poly), replace=False)]
        #     if target:
        #         m["p_target"] = points[rng.choice(np.delete(np.arange(25), poly), replace=False)]
        #     m["boundary_points"] = points[poly]

        # Rectangle
        x1,x2,y1,y2 = 0.2*rng.rand(4)
        x2 = w - x2
        y2 = h - y2
        m["boundary_points"] = np.array([[x1,y1],[x2,y1],[x2,y2], [x1,y2]])
        x_s, x_t = x1 + 1.5*D_MIN + rng.rand(2)*(x2-x1-3*D_MIN)
        y_s, y_t = y1 + 1.5*D_MIN + rng.rand(2)*(y2-y1-3*D_MIN)
    else:
        x_s, x_t = 0.05*w + rng.rand(2)*0.9*w
        y_s, y_t = 0.05*h + rng.rand(2)*0.9*h
    m["p_start"] = [x_s,y_s]

    if target:
        m["p_target"] = [x_t,y_t]
        # to ensure uniform distribution of target ratios
        r = rng.choice(RATIOS)
        r *= [-1,1][rng.rand()<0.5] # + or -
        m["i_target"]= r

    return m


def pol2car(r,phi):
    """ Transforms polar coordinates to cartesian ones (x,y)"""
    return r*np.cos(phi), r*np.sin(phi)



WIDTH = HEIGHT = 1
D_MAX = .8
D_MIN = 0.05
I_MAX = 5
EPS = 0.05 # tolerance for position
I_EPS = 0.2 # tolerance for ratios
RATIOS = get_admisible_ratios(I_MAX)
MAX_TRAJ_LEN = 8


class StageCreator(GoalEnv):
    def __init__(self, N, ae=None, env_map=None, boundary=0.3, target=True, seed=None, mode="goal"):
        """ Goal based environment, contained to the design of as single gearbox stage (2D task) i.e.
            finding a chain of gears that will connect the starting and target points while
            achieving desired ratio.
            It supports two modes: "goal" or "selfplay" in the latter no target is created, instead
            it will be set by the agent itself, also the agant is collision immune.
            Params:
            - env_map: a vector [x_start, y_start, x_target, y_target, i_target] of
                normalized cartesian coordinates [0,1] and desired ratio. If None, a random
                map will be created
            - boundary is the probability of creating an environemnt with boundary
            - target: when creating an environemnt, sample a random target or not (relevent for selfplay)
            - seed may be used to have the pseudo random maps always appear in the same order.
        """

        self.seed(seed)
        self.boundary=boundary
        self.screen = None
        self.fig, self.ax = plt.subplots(1,3, figsize=[16,5])
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # For screen output
        self.N = N # resolution
        self.h_x = WIDTH/N
        self.h_y = HEIGHT/N
        self.screen = np.zeros((N,N))
        self.ae = ae # output reshaped and normalized features
        
        if ae == None:
            obs_space = spaces.Tuple((spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=float), 
                                    spaces.Box(low=np.array([0,0,-I_MAX,D_MIN]), high=np.array([1,1,I_MAX,D_MAX])) ))
        else:
            out = 4 + ae.get_bottleneck_size(N)[-1] # x, y, i ,diam + hidden dim
            obs_space = spaces.Box(low=-1.0, high=1.0, shape=(out,))
        
        self.observation_space = spaces.Dict({
                    'observation': obs_space, 
                    'desired_goal': spaces.Box(low=np.array([0,0,-I_MAX]), high=np.array([1,1,I_MAX])), # x_target, y_target, i_target
                    'achieved_goal': spaces.Box(low=np.array([0,0,-I_MAX]), high=np.array([1,1,I_MAX]))}) # x_current, y_current, i_current

        # action is a tuple (diam, phi)
        self.action_space = spaces.Box(low=np.array([D_MIN,0]), high=np.array([D_MAX, 1]))

        self.reset(random=True, env_map=env_map, target = mode=="goal")


    # def step(self, a:tuple):
    #     """ Irespectful of the mode returns a tuple: (obs, done, reward, info)
    #         - a is a tuple of (diam, phi) or (diam, phi, agent_idx) where 
    #         agent_idx is necessary to distinguish who is playing in the selfplay
    #         mode.
    #         In selfplay mode there is no termination upon collision, instead 
    #         just no change in state. Also, the rewards are calculated externally 
    #         so they don't matter in that case.
    #     """
    #     reward = 0
    #     next_state = self.state.copy()
    #     old_d = next_state[3]
    #     done = False
    #     self.steps += 1 # just for the recprd

    #     # First step doesn't change the position because one gear doesn't make
    #     # a gear stage (every step modifies d though)
    #     next_state[3] = a[0]
    #     if self.traj!=[]:
    #         next_state[:2] += pol2car((old_d+a[0])/2, 2*np.pi*a[1]) # new position
    #         next_state[2] *= -old_d/a[0] # new ratio
    #     if a[0]<D_MIN or len(self.traj)>MAX_TRAJ_LEN or self.check_collisions(*next_state[[0,1,3]]):
    #         if self.mode=="goal":
    #             done = True  # no termination during selfplay
    #             self.state = obs = next_state
    #             self.traj.append(tuple(next_state[[0,1,3]])) # using tuple it will be appended by value          
    #         else:
    #             obs = self.state.copy() # state remains unchanged
    #     else:
    #         self.state = obs = next_state
    #         self.traj.append(tuple(next_state[[0,1,3]])) # using tuple it will be appended by value
    #         # reward is ignored in selfplay mode, but it may not be calulated by task giver
    #         # because its goal is equal to state_init
    #         if not (self.mode=="selfplay" and a[-1]==1):
    #             reward, done = self.compute_reward(next_state, self.goal, None)

    #     return self.observation(obs), reward, done, {}

    def step(self, a:tuple):
        """ Irespectful of the mode returns a tuple: (obs, done, reward, info)
            - a is a tuple of (diam, phi) or (diam, phi, agent_idx) where 
            agent_idx is necessary to distinguish who is playing in the selfplay
            mode.
            In selfplay mode there is no termination upon collision, instead 
            just no change in state. Also, the rewards are calculated externally 
            so they don't matter in that case.
        """
        reward = 0
        next_state = self.state.copy()
        old_d = next_state[3]
        done = False
        info = {}
        self.steps += 1 # just for the record

        # First step doesn't change the position because one gear doesn't make
        # a gear stage (every step modifies d though)
        next_state[3] = a[0]
        if self.traj!=[]:
            next_state[:2] += pol2car((old_d+a[0])/2, 2*np.pi*a[1]) # new position
            next_state[2] *= -a[0]/old_d # new ratio
        if a[0]<D_MIN or self.check_collisions(*next_state[[0,1,3]]):
            if self.mode=="goal" or len(self.traj)>MAX_TRAJ_LEN or self.steps>15:
                done = True  
                self.state = obs = next_state
                self.traj.append(tuple(next_state[[0,1,3]])) # using tuple it will be appended by value          
            else: # no termination during selfplay
                next_state = obs = self.state.copy() # state remains unchanged
                info["invalid"] = True # = this move would have caused termination
        else:
            self.state = obs = next_state
            self.traj.append(tuple(next_state[[0,1,3]]))
            reward, done = self.compute_reward(next_state, self.goal, None)

        return self.observation(obs), reward, done, info


    def compute_reward(self, state, target, info):
        """Compute the sparse binary reward (according to gym.GoalEnv interface):
            - = 1: if the target position has been met within EPS and
                the rotation direction is same
            - = 0: otherwise
        """
        reward = 0
        done = False
        p_dist = np.linalg.norm(state[:2]-target[:2], 2)
        if p_dist<self.state[3]/2: # if the gear occludes the output
            done = True
            # if p_dist<EPS: # .. if it does so within the tolerance
            i_ratio = target[2]/state[2] # ratio of ratios
            # reward = int(i_ratio>0) # same sign?
            if i_ratio>0 and abs(target[2] - state[2])<I_EPS:
                reward = 1
        return reward, done


    def reset(self, random=True, env_map=None, target=True):
        """ Initializes the environment. If env_map==None and random==True a random 
            map w/ or w/0 a target (relevant for selfplay mode) will be created. If 
            random==False the environment will be reset to the last initial state.
        """
        try:
            if env_map!=None:
                assert isinstance(env_map, dict)
            elif random:
                env_map = generate_random_map(self.np_random, WIDTH, HEIGHT, self.boundary, target)
                print(env_map)
            self.b_points = env_map.get("boundary_points", None)
            self.state_init = [*env_map["p_start"], 1.0, 0] # x,y,i,d
            # if target==False (e.g in selfplay mode), the target and init will be same
            self.goal = np.array([*env_map.get("p_target", env_map["p_start"]), 
                                    env_map.get("i_target", 1.0)])
        except TypeError: # if random==False and env_map==None -> no env_map variable
            pass # will ignore the assignemt of goal and state_init and just use their old values
        except AttributeError: # triggers if env_map is None
            pass      

        # reset state to initial (goal and b_points always stay the same)
        self.state = obs = np.array(self.state_init)
        self.traj = []
        self.steps = 0 # for selfplay mode
        self._init_fig()

        return self.observation(obs)


    def observation(self, obs):
        if self.traj != []:
            x,y,d = self.traj[-1]
            self._draw_circle(x,y,d)

        if self.ae == None:
            return {'achieved_goal':self.state[[0,1,2]], 
                    'desired_goal':self.goal, 
                    'observation': (np.array([self.screen], dtype=np.float32), obs)}
            # return np.array([self.screen], dtype=np.float32), obs
        else:
            screen_t = torch.FloatTensor([self.screen]).to(self.device).unsqueeze(0)
            # features from pixel input, reshaped to a vector and normalized
            features = self.ae(screen_t).squeeze(0).detach().cpu().numpy()
            obs = np.hstack((features, obs))
            return {'achieved_goal':self.state[[0,1,2]], 
                    'desired_goal':self.goal, 
                    'observation':obs}
            # return obs


    def render(self, mode="human", delay=0.1, path=None):
        """ Displays the state trajectory of the environment up to the
            current state. Modes: 
            - "human" - just plots the fig
            - "rgb_array" - returns the cavas as a np.array
        """
        if self.traj != []:
            n = 50
            k = np.linspace(0,1,n)
            x_c, y_c, _, d = self.state
            x,y = pol2car([d/2]*n, 2*np.pi*k)
            self._draw_circle(*self.traj[-1]) 
            self.ax[0].plot(np.array(x)+x_c, np.array(y)+y_c, c="tab:blue")  
                     

        if mode=="human":        
            self.ax[0].set_title(f"Step: {self.steps}"
                                +f" i = {self.state[-2]}" 
                                +f" i_target = {self.goal[-1] :.3f}")

            self.ax[1].pcolormesh(self.screen, cmap="binary") # screen output
            if self.ae!=None: # screen passed throughthe autoencoder
                out = torch.FloatTensor([[self.screen]]).to(self.device)
                out = self.ae.forward(out).detach().squeeze(0).numpy()
                self.ax[2].pcolormesh(out[0], cmap="binary")

            self.fig.legend()
            plt.pause(delay)
            if path!=None:
               plt.savefig(path)
        elif mode=="rgb_array":
            # Save canvas to a numpy array.
            data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
        else:
            Exception("Unsupported mode")


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed


    def _init_fig(self):
        """ Plots starting and end points as well as boundary """
        plt.cla()
        self.ax[0].clear()
        self.ax[0].set_xlim([0, 1])
        self.ax[0].set_ylim([0, 1])
        self.ax[0].scatter(*self.goal[:2], c="tab:red", label="Target")
        self.ax[0].scatter(*self.state[:2], c="tab:green", label="Input")

        # Close the polygon and plot the boundary
        try:
            v_x,v_y = list(zip(*self.b_points))
            self.ax[0].plot(list(v_x) +[v_x[0]], list(v_y) + [v_y[0]], c='k')
        except:
            pass
        
        # For screen output
        self.screen = np.zeros_like(self.screen)
        self._draw_boundary()

        # self.fig.canvas.draw() # for rgb_mode rendring


    def set_target(self, p_target, i_target):
        self.goal = [*p_target, i_target]


    def check_collisions(self, x, y, d):
        """ Checks whether the newly added gear doesn't collide
            with any of the previous ones or the map boundaries/obstacles
        """
        # with the map
        try:
            i=-1
            for j in range(len(self.b_points)):
                p1 = self.b_points[i,:]
                p2 = self.b_points[j,:]
                # a = (p2[1]-p1[1])/(p1[0]-p2[0])
                # dist = y + a*x - p2[1] - a*p2[0]
                dist = (p2[0]-p1[0])*(p1[1]-y) - (p1[0]-x)*(p2[1]-p1[1])
                dist = abs(dist)/np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                if dist<(d/2):
                    return True
                i=j
        except: # no boundary case
            if x+d/2> WIDTH or y+d/2> HEIGHT or x-d/2 <0 or y-d/2<0:
                return True

        # with itself
        for x_i,y_i,d_i in self.traj:
            if (x_i-x)**2 + (y_i-y)**2 - (d_i+d)**2/4 < -1e-6:
                return True

        return False


    def _draw_boundary(self):
        """ Uses ray casting a.k.a even-odd rule to determine
            wheter a point lies within the boundary points or not
        """
        try:
            if len(self.b_points)==4:
                # b_points contain [[x1,y1],[x2,y1],[x2,y2], [x1,y2]]
                self.screen.fill(1)
                [i1,j1], [i2,j2] = [[int(x//self.h_x), int(y//self.h_y)] for [x,y] in self.b_points[[0,2]]]
                # print(i1,i2,j1,j2)
                self.screen[j1:j2+1, i1:i2+1] = 0
            else:    
                for j in range(self.N):
                    for i in range(self.N):
                        x = (i+0.5)*self.h_x
                        y = (j+0.5)*self.h_y
                        c = 1
                        k=-1
                        for l in range(len(self.b_points)):
                            p1 = self.b_points[k,:]
                            p2 = self.b_points[l,:]
                            crs = ((p1[1]>y) != (p2[1]>y)) and \
                                (x<p1[0] + (p2[0]-p1[0])*(y-p1[1])/(p2[1]-p1[1]))
                            c = (c+int(crs))%2
                            k=l
                        self.screen[j,i] = c
        except:
            pass


    def _draw_circle(self,x,y,d):
        """ Draws pixelated circle centered at x,y with diamter d. 
            Takes into account the point's position within the cell. 
        """
        N = self.N # resolution
        # Determine the boundary cell indices (min & max to draw even if the circle is too big)
        y_u = min(int((y+d/2)*N+0.5), N) # 0.5 is because int() is essentially floor operator
        y_l = max(int((y-d/2)*N+0.5),0)
        x_r = min(int((x+d/2)*N+0.5), N)
        x_l = max(int((x-d/2)*N+0.5),0)

        inside = lambda x_i,y_i: np.sqrt(((y_i+.5)/N-y)**2 + ((x_i+.5)/N-x)**2)<d/2
        for y_idx in range(y_l,y_u): # range is [) !
            for x_idx in range(x_l,x_r):
                if inside(x_idx, y_idx):
                    self.screen[y_idx, x_idx] = 1




if __name__=="__main__":
    from Design.Models.AE.model import Autoencoder84
    RES = 84 # resolution for the screen output

    device = torch.device("cpu")
    ae = Autoencoder84(1, pretrained="./Design/Models/AE/Autoencoder84.dat").to(device).float()
    
    env = StageCreator(RES, ae=ae, mode="goal", boundary=.8, seed=12345678)
    env.mode="selfplay" # to have the first target generated

    print(f"Obs spaces: {env.observation_space.spaces}")
    print(isinstance(env, GoalEnv))
    print(env)

    print(f"Target ratio:{env.goal[-1]}")
    print(f"Current ratio:{env.state[2]}")

    obs = env.reset(random=False)
    # print(obs)
    print(f"ObservationSpace contains obs: {env.observation_space.contains(obs)}")
    env.render()
    # play manually
    flag = True
    while flag:
        d,phi = input("Action (d [D_MIN,DMAX], phi [0,1]) or q to quit:").split()
        new_obs, reward, done, _ = env.step((float(d),float(phi)))

        # print("Old/New obs difference", obs["observation"]-new_obs["observation"])
        obs = new_obs.copy()
        print(f"Target ratio:{env.goal[-1]}")
        print(f"Current ratio:{env.state[2]}")
        env.render()
        if done:
            print(f"Reward: {reward}")
            a = input("Start again with new env (y/n)?: \n")
            flag = a=='y'
            env.reset(random=False)
            env.render()

    plt.close(fig='all')