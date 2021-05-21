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

from lib import model


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


def generate_random_map(rng, w, h, boundary=0.2):
    """ A map is a vector [x_start, y_start, x_target, y_target, i_target] i.e. 
        position of start and finish as well as target ratio (i_start=1).
        Additionaly a random boundary might be created
        The csys for a map is given in the lower left corner so that all coordinates
        are positive
        boundary: probability of creating a map with boundary
    """
    m ={}
    # Boundary polygon
    if rng.rand()<boundary:
        # Generate a number of random points in [0,1]*w x [0,1]*h
        points = rng.rand(25,2)*[w,h]
        hull = ConvexHull(points)
        # Chose start and target points from the points within the hull
        idx=rng.choice(np.delete(np.arange(25), hull.vertices),2)
        (x_s,y_s), (x_t,y_t) = points[idx,:]
        m["boundary_points"] = points[hull.vertices,:]
    else:
        x_s, x_t = 0.05*w + rng.rand(2)*0.9*w
        y_s, y_t = 0.05*h + rng.rand(2)*0.9*h
    m["p_start"] = [x_s,y_s]
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
I_MAX = 10
EPS = 0.05
RATIOS = get_admisible_ratios(I_MAX)


class StageCreator(Env):
    def __init__(self, env_map=None, boundary=0.3, seed=None):
        """ - env_map is a vector [x_start, y_start, x_target, y_target, i_target],
            - seed may be used to have the pseudo random maps always appear 
                in the same order.
            - static environments always use the same map
        """
        self.seed(seed)
        self.boundary=boundary
        env_map = generate_random_map(self.np_random, WIDTH, HEIGHT,boundary) if env_map==None else env_map
        self.i_target = env_map["i_target"]
        self.p_target = env_map["p_target"] # target position
        self.p_start = env_map["p_start"]
        self.b_points = env_map.get("boundary_points",None) # boundary points
        self.screen = None

        # state appended with the (x_current, y_current, i_current, x_target, y_target, i_target)
        self.observation_space = spaces.Box(low=np.array([0,0,-100,0,0,-100]), high=np.array([1,1,100,1,1,100]), dtype=np.float)
        # action is a tuple (diam, phi)
        self.action_space = spaces.Box(low=np.array([D_MIN,0]), high=np.array([D_MAX, 1]), dtype=np.float)

        self.fig, self.ax = plt.subplots(1,3, figsize=[15,5])
        self.reset(random=False)


    def step(self, a:tuple):
        """ Termination if:
            - |output - target|<=EPS -> reward 1
            - collision -> reward -1
        """
        done = False
        reward=0
        next_state = self.s.copy()
        if a[0]<D_MIN:
            # return self.s, -1, True, None
            return next_state, reward, True, None
        if self.traj==[]:
            done = self.check_collisions(*next_state[:2], a[0])
            self.traj.append((*self.p_start, a[0]))
            # reward = -int(done)
        else:
            d_old = self.traj[-1][-1]
            # print("Pre update: ", self.s)
            next_state[:2] += pol2car((d_old+a[0])/2, 2*np.pi*a[1]) # new position
            next_state[2] *= -d_old/a[0] # new ratio
            done = self.check_collisions(*next_state[:2], a[0])
            self.traj.append((*next_state[:2], a[0]))
            # print("Post update: ", self.s)
            # reward = -int(done)
        
        p_dist = np.linalg.norm(next_state[:2]-next_state[-3:-1], 2)
        if p_dist<(a[0]+D_MIN)/2: # if the gear occludes the output
            done = True
            if p_dist<D_MIN/2:
                i_ratio = next_state[-1]/next_state[2]
                if i_ratio < 0:
                    # reward = -1
                    reward = 0
                elif abs(np.log(i_ratio))<0.05:
                    reward = len(self.traj)>1
                else:
                    reward = 0
            # i_ratio = self.s[-1]/self.s[2]
            # if i_ratio < 0:
            #     reward = -1
            # else:
            #     reward = max(0,1-abs(np.log(i_ratio)))

        self.s = next_state
        
        return next_state, reward, done, None


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

        self.ax[0].set_title(f"Current ratio: {self.s[2]}, Target ratio:{self.i_target}")
        self.fig.legend()
        plt.pause(delay)
        # if path is not None:
        #     plt.savefig(path + f"stage_{self.stage_no}")


    def reset(self, random=True):
        if random:
            env_map = generate_random_map(self.np_random, WIDTH, HEIGHT, self.boundary)
            self.i_target = env_map["i_target"]
            self.p_target = env_map["p_target"] # target position
            self.p_start = env_map["p_start"]
            self.b_points = env_map.get("boundary_points",None)
        self.s = np.array([*self.p_start, 1.0, *self.p_target, self.i_target])
        self.traj = []

        plt.cla()
        self.ax[0].clear()
        self.ax[0].set_xlim([0, 1])
        self.ax[0].set_ylim([0, 1])
        self.ax[0].scatter(*self.p_start, c="tab:green", label="Input")
        self.ax[0].scatter(*self.p_target, c="tab:red", label="Target")
        # Close the polygon and plot the boundary
        if self.b_points is not None:
            v_x,v_y = list(zip(*self.b_points))
            self.ax[0].plot(list(v_x) +[v_x[0]], list(v_y) + [v_y[0]], c='k')

        return self.s.copy()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed


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
        self.observation_space = spaces.Tuple((spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=np.float32), self.observation_space))
        self.screen = np.zeros((N,N))
        # self.observation_space =  spaces.Box(low=0.0, high=1.0, shape=(1,N,N), dtype=np.float32)
        
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
        return np.array([self.screen.copy()], dtype=np.float32), obs


    # def reset(self):
    #     _, obs = super().reset()
    #     self.screen = np.zeros_like(self.screen)
    #     # self._draw_io()
    #     if self.b_points!=None:
    #         self._draw_boundary()

    #     return np.array([self.screen], dtype=np.float32), obs

    
    def render(self, delay=0.1, ae=None):
        super().render()
        self.ax[1].pcolormesh(self.screen, cmap="binary")
        if ae!=None:
            inp = np.array([[self.screen]])
            out = ae(torch.tensor(inp).float()).detach().numpy()
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


    # def _draw_boundary(self, quadrant:int, poly:list, level=3):
    #     """ Quadtree, recursive implementation of a ray casting 
    #         a.k.a even-odd rule to determine 
    #         wheter a point lies within the boundary hull or not
    #     """
    #     if level==0:
    #         pass
    #     else:
    #         q = poly//0.5

    #         self._draw_boundary(level-1,0,)
    #         # Check which quadrants intersect with poly


    #     poly = self.hull.points[self.hull.simplices]

    #     for j in range(self.N):
    #         for i in range(self.N):
    #             x = (i+0.5)*self.h_x
    #             y = (j+0.5)*self.h_y
    #             c = 1
    #             for p1,p2 in poly:
    #                 crs = ((p1[1]>y) != (p2[1]>y)) and \
    #                     (x<p1[0] + (p2[0]-p1[0])*(y-p1[1])/(p2[1]-p1[1]))
    #                 c = (c+int(crs))%2
    #             self.screen[j,i] = c
    
    # @staticmethod
    # def _check_intersections(quadrant, poly):
    #     """ Returns a list of tuples of indices of subquadrants 
    #         in which intersection with the polygon occurs and 
    #         the portion of the polygon responsible.
    #         Polygon vertices are relative in to the quandrant's csys
    #         Quadrants:      ,,,,,,,,,
    #                         | 1 | 3 |
    #                         |---+---|
    #                         | 0 | 2 |
    #                         '''''''''
    #     """
    #     # First check vertices
    #     q = np.dot(poly//0.5,[2,1]).astype(int) # quadrants
    #     # Extract information on relevant points for each quadrant
    #     i=-1
    #     j=0
    #     quadrants=[[],[],[],[]]
    #     for k in [0,2,3,1]:
    #         while k==q[j]:
    #             quadrants[k].append(poly[j])
    #             j += 1
    #         q[i]
    #         q[j]

    #     for j in range(len(q)):
    #         quadrant = q[j]
    #         if q[j]!=q[i]



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


    # def _draw_circle(self,x,y,d):
    #     """ Draws pixelated circles. 
    #         Doesn't take point's position within the cell into account
    #     """
    #     N = self.N
    #     # First y range (min & max to draw even if the circle is too big)
    #     y_u = min(int((y+d/2+1/(2*N))*N), N)
    #     y_l = max(int((y-d/2+1/(2*N))*N),0)
    #     x_c, y_c = (np.array([x,y])*N).astype(int)
    #     res = (y+d/2)%(HEIGHT/N)
    #     for y_idx in range(y_l,y_u): # range is [) !
    #         # what is the x range at particular height y_idx + res
    #         if y_idx<y_c:
    #             dx = d/2*np.cos(np.arcsin(max(-1,2*((y_idx-1)/N + res - y)/d)))
    #         else:
    #             dx = d/2*np.cos(np.arcsin(min(1,2*(y_idx/N + res - y)/d)))
    #         # x range: x_left, x_right
    #         x_l = max(int((x-dx+1/(2*N))*N), 0)
    #         x_r = min(int((x+dx+1/(2*N))*N), N)
    #         # x_r +=1 if x_l==x_r else 0
    #         self.screen[y_idx, x_l:x_r] = 1 # array indexeing is also [)


# class LazyFrames(object):
#     def __init__(self, frames):
#         """This object ensures that common frames between the observations are only stored once.
#         It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
#         buffers.
#         This object should only be converted to numpy array before being passed to the model.
#         You'd not belive how complex the previous solution was."""
#         self._frames = frames

#     def __array__(self, dtype=None):
#         out = np.concatenate(self._frames, axis=0)
#         if dtype is not None:
#             out = out.astype(dtype)
#         return out


# class StackFrames(Wrapper):
#     """Stacks k last frames, returns lazy array for efficiency
#         see: baselines.common.atari_wrappers.LazyFrames
#     """
#     def __init__(self, env, k):
#         super.__init__(self, env)
#         self.k = k
#         self.frames = deque([], maxlen=k)
#         shp = env.observation_space.shape
#         self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.float32)

#     def reset(self):
#         """ Initial stack just contains k initial frames"""
#         obs = self.env.reset()
#         for _ in range(self.k):
#             self.frames.append(obs)
#         return self._get_ob()

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         self.frames.append(obs)
#         return self._get_ob(), reward, done, info

#     def _get_ob(self):
#         assert len(self.frames) == self.k
#         return LazyFrames(list(self.frames))


# class AppendGoal(ObservationWrapper):
#     def __init__(self, env=None):
#         super.__init__(env)
#         goal = spaces.Box(low=np.array([0,0,-100]), high=np.array([1,1,100]), dtype=np.float)
#         self.observation_space = spaces.Tuple((self.observation_space, goal))
    
#     def observation(self, obs):
#         return obs, *self.p_target, self.i_traget




if __name__=="__main__":
    device = torch.device("cpu")
    env = StageCreator(boundary=.8)
    env = ScreenOutput(128, env)

    ae = model.Autoencoder(1, pretrained="./Autoencoder-FC.dat").to(device).float()
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
