import sys
import os
sys.path.append(os.getcwd())

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from fractions import Fraction
import itertools as it
from scipy.spatial import Delaunay, ConvexHull
import torch

from gym import Env, spaces, ObservationWrapper, Wrapper
from gym.utils import seeding

import model


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
RES = 64 # resolution for the screen output
D_MAX = 0.7
D_MIN = 0.05
I_MAX = 5
EPS = 0.05
RATIOS = get_admisible_ratios(I_MAX)


class StageCreator(Env):
    def __init__(self, env_map=None, boundary=0.3, target=False, seed=None):
        """ - env_map is a vector [x_start, y_start, x_target, y_target, i_target],
            - seed may be used to have the pseudo random maps always appear 
                in the same order.
            - static environments always use the same map
        """
        self.seed(seed)
        self.boundary=boundary
        env_map = generate_random_map(self.np_random, WIDTH, HEIGHT, boundary, target) if env_map==None else env_map
        self.i_target = env_map.get("i_target", 1.0)
        self.p_start = env_map["p_start"]
        self.p_target = env_map.get("p_target", self.p_start.copy())
        self.b_points = env_map.get("boundary_points",None) # boundary points
        self.screen = None
        self.steps = 0 # for the collision free step()

        # state appended with the (x_current, y_current, i_current, x_target, y_target, i_target)
        self.observation_space = spaces.Box(low=np.array([0,0,-100,0,0,-100]), high=np.array([1,1,100,1,1,100]), dtype=float)
        # action is a tuple (diam, phi)
        self.action_space = spaces.Box(low=np.array([D_MIN,0]), high=np.array([D_MAX, 1]), dtype=float)

        self.fig, self.ax = plt.subplots(1,3, figsize=[15,5])
        self.reset(random=False)


    # def step(self, a:tuple):
    #     """ Termination if:
    #         - |output - target|<=EPS -> reward 1
    #         - collision -> reward -1
    #     """
    #     done = False
    #     reward=0
    #     next_state = self.s.copy()
    #     if a[0]<D_MIN or len(self.traj)>8:
    #         # return self.s, -1, True, None
    #         if self.traj!=[]:
    #             # Update the next state nevertheless
    #             d_old = self.traj[-1][-1]
    #             next_state[:2] += pol2car((d_old+a[0])/2, 2*np.pi*a[1]) # new position
    #             next_state[2] *= -d_old/a[0] # new ratio
    #         return next_state, reward, True, None
    #     if self.traj==[]:
    #         # First step doesn't change the state because one gear doesn't make
    #         # a gear stage
    #         done = self.check_collisions(*next_state[:2], a[0])
    #         self.traj.append((*self.p_start, a[0]))
    #         # reward = -int(done)
    #     else:
    #         d_old = self.traj[-1][-1]
    #         # print("Pre update: ", self.s)
    #         next_state[:2] += pol2car((d_old+a[0])/2, 2*np.pi*a[1]) # new position
    #         next_state[2] *= -d_old/a[0] # new ratio
    #         done = self.check_collisions(*next_state[:2], a[0])
    #         self.traj.append((*next_state[:2], a[0]))
    #         # print("Post update: ", self.s)
    #         # reward = -int(done)
        
    #     p_dist = np.linalg.norm(next_state[:2]-next_state[-3:-1], 2)
    #     if p_dist<(a[0]+D_MIN)/2: # if the gear occludes the output
    #         done = True
    #         if p_dist<D_MIN/2:
    #             i_ratio = next_state[-1]/next_state[2]
    #             if i_ratio < 0:
    #                 # reward = -1
    #                 reward = 0
    #             elif abs(np.log(i_ratio))<0.4:
    #                 reward = len(self.traj)>1
    #             else:
    #                 reward = 0
    #         # i_ratio = self.s[-1]/self.s[2]
    #         # if i_ratio < 0:
    #         #     reward = -1
    #         # else:
    #         #     reward = max(0,1-abs(np.log(i_ratio)))

    #     self.s = next_state
        
    #     return next_state, reward, done, None

    def step(self, a:tuple):
        """ No termination upon collision, instead just no change in state,
            Termination upon reaching max steps or reaching the target
            the number of steps is ultimately used to calculate reward 
            Returns a tuple: next_state, done
        """
        next_state = self.s.copy()
        self.steps += 1

        if a[0]<D_MIN:
            return next_state, None, False, None # invalid action
        if self.traj!=[]: # If at least one gear already placed
            d_old = self.traj[-1][-1]
            next_state[:2] += pol2car((d_old+a[0])/2, 2*np.pi*a[1]) # new position
            next_state[2] *= -d_old/a[0] # new ratio
        if self.check_collisions(*next_state[:2], a[0]):
            return self.s.copy(), None, False, None # return old state
        self.traj.append((*next_state[:2], a[0]))
        self.s = next_state
        
        if a[-1]==0: # if Bob is playing
            p_dist = np.linalg.norm(next_state[:2]-next_state[-3:-1], 2)
            if p_dist<D_MIN:
                if next_state[2]*next_state[-1]<0: # opposite rotation direction
                    return next_state, None, False, None
                if abs(next_state[2]-next_state[-1])<EPS:
                    return next_state, None, True, None # Succesfully finished

        return next_state, None, False, None


    def render(self, mode="human", delay=0.1, path=None, ae=None):
        """ Displays the state trajectory of the environment up to the
            current state. Works currently for 2D case only.
            mode: "human" or "screen"
        """
        if self.traj != []:
            n = 50
            k = np.linspace(0,1,n)
            x_c,y_c,d = self.traj[-1]
            x,y = pol2car([d/2]*n, 2*np.pi*k)
            self.ax[0].plot(np.array(x)+x_c, np.array(y)+y_c, c="tab:blue")            

        # self.ax[0].set_title(f"It: {len(self.traj)}, Current ratio: {self.s[2]}, Target ratio:{self.i_target}")
        self.ax[0].set_title(f"Step: {self.steps}, Current ratio: {self.s[2]}")
        self.fig.legend()
        plt.pause(delay)
        # if path is not None:
        #     plt.savefig(path + f"stage_{self.stage_no}")


    def reset(self, random=True, target=False):
        if random:
            env_map = generate_random_map(self.np_random, WIDTH, HEIGHT, self.boundary, target)
            self.i_target = env_map.get("i_target", 1.0)
            self.p_start = env_map["p_start"]
            self.p_target = env_map.get("p_target", self.p_start.copy())
            self.b_points = env_map.get("boundary_points",None)
        self.s = np.array([*self.p_start, 1.0, *self.p_target, self.i_target])
        self.traj = []
        self.steps = 0

        plt.cla()
        self.ax[0].clear()
        self.ax[0].set_xlim([0, 1])
        self.ax[0].set_ylim([0, 1])
        self.ax[0].scatter(*self.p_target, c="tab:red", label="Target")
        self.ax[0].scatter(*self.p_start, c="tab:green", label="Input")

        # Close the polygon and plot the boundary
        if self.b_points is not None:
            v_x,v_y = list(zip(*self.b_points))
            self.ax[0].plot(list(v_x) +[v_x[0]], list(v_y) + [v_y[0]], c='k')

        return self.s.copy()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def set_target(self, p_target, i_target):
        self.p_target = p_target
        self.i_target = i_target


    def check_collisions(self, x, y, d):
        """ Checks whether the newly added gear doesn't collide
            with any of the previous ones or the map boundaries/obstacles
        """
        # with the map
        if self.b_points is not None:
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
        else: 
            if x+d/2> WIDTH or y+d/2> HEIGHT or x-d/2 <0 or y-d/2<0:
                return True

        # with itself
        for x_i,y_i,d_i in self.traj:
            if (x_i-x)**2 + (y_i-y)**2 - (d_i+d)**2/4 < -1e-6:
                return True

        return False


class ScreenOutput(ObservationWrapper):
    """ Extends the state of the environment by a pixel output of the entire gear stage
        created do far.
        Returns a tuple of single channel, gray-scale NxN pixel picture in a format required 
        by Pytorch and the oryginal state (x_current, y_current, i_current, x_target, y_target, i_target)
    """
    def __init__(self, N, env=None):
        super().__init__(env)
        self.N = N # resolution
        self.h_x = WIDTH/N
        self.h_y = HEIGHT/N
        self.observation_space = spaces.Tuple((spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=float), self.observation_space))
        self.screen = np.zeros((N,N))
        # self.observation_space =  spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=float)
        
        # self._draw_io()
        if self.b_points is not None:
            self._draw_boundary()


    def observation(self, obs):
        if self.traj != []:
            x,y,d = self.traj[-1]
            N = self.N # resolution
            self._draw_circle(x,y,d)
            # # recolor the input if ocluded
            # if len(self.traj) == 1: 
            #     h,v = np.array([-1,0,0,0,1]),np.array([0,-1,0,1,0])
            #     self.screen[h+y_c,v+x_c] = 0.587 #green
        else: # trigers at reset
            self.screen = np.zeros_like(self.screen)
            # self._draw_io()
            if self.b_points is not None:
                self._draw_boundary()

        # print("Wrapper obs:", obs)
        return np.array([self.screen.copy()], dtype=float), obs


    # def reset(self):
    #     _, obs = super().reset()
    #     self.screen = np.zeros_like(self.screen)
    #     # self._draw_io()
    #     if self.b_points!=None:
    #         self._draw_boundary()

    #     return np.array([self.screen], dtype=float), obs

    
    def render(self, delay=0.1, ae=None):
        super().render()
        self.ax[1].pcolormesh(self.screen, cmap="binary")
        if ae!=None:
            inp = np.array([[self.screen]])
            out = ae.forward(torch.tensor(inp).float()).detach().numpy()
            self.ax[2].pcolormesh(out[0][0], cmap="binary")
        plt.pause(delay)


    def _draw_io(self):
        """ Draw small crosses in place of input and output """
        x_s, y_s = (np.array(self.p_start)*self.N).astype(int)
        x_t, y_t = (np.array(self.p_target)*self.N).astype(int)
        h,v = np.array([-1,0,0,0,1]),np.array([0,-1,0,1,0])
        self.screen[h+y_s,v+x_s] = 0.587 #green
        self.screen[h+y_t,v+x_t] = 0.299 #red


    def _draw_boundary(self):
        """ Uses ray casting a.k.a even-odd rule to determine
            wheter a point lies within the boundary points or not
        """
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


    def _draw_circle(self,x,y,d):
        """ Draws pixelated circles. 
            Takes into account the point's position within the cell 
        """
        N = self.N # resolution
        # First y range (min & max to draw even if the circle is too big)
        y_u = min(int((y+d/2)*N+0.5), N)
        y_l = max(int((y-d/2)*N+0.5),0)
        x_c, y_c = (np.array([x,y])*N).astype(int)
        res = (y+d/2)%self.h_y # "residual"
        for y_idx in range(y_l,y_u): # range is [) !
            # what is the x range at particular height y_idx + res
            if y_idx<y_c:
                dx = d/2*np.cos(np.arcsin(max(-1,2*((y_idx-1)/N + res - y)/d)))
            elif y_idx>y_c:
                dx = d/2*np.cos(np.arcsin(min(1,2*(y_idx/N + res - y)/d)))
            else:
                dx = d/2
            # x range: x_left, x_right
            x_l = max(int((x-dx)*N+0.5), 0)
            x_r = min(int((x+dx)*N+0.5), N)
            # x_r +=1 if x_l==x_r else 0
            self.screen[y_idx, x_l:x_r] = 1 # array indexeing is also [)


if __name__=="__main__":
    device = torch.device("cpu")
    env = StageCreator(boundary=.8, target=True)
    env = ScreenOutput(128, env)

    ae = model.Autoencoder(1, pretrained="./Design/Models/DDPG/Autoencoder-FC.dat").to(device).float()
    # print(ae.get_fe_out_size((1,RES,RES)))
    # ae=None
    env.render(ae=ae)

    print(f"Target ratio:{env.s[-1]}")
    print(f"Current ratio:{env.s[2]}")

    # play manually
    flag = True
    while flag:
        d,phi = input("Action (d [D_MIN,DMAX], phi [0,1]) or q to quit:").split()
        new_state, reward, done, _ = env.step((float(d),float(phi)))
        # new_state, done = env.step((float(d),float(phi)))
        
        print(f"Target ratio:{env.s[-1]}")
        print(f"Current ratio:{env.s[2]}")
        env.render(ae=ae)
        if done:
            print(f"Reward: {reward}")
            a = input("Start again with new env (y/n)?: \n")
            flag = a=='y'
            env.reset()
            env.render(ae=ae)

    plt.close(fig='all')
