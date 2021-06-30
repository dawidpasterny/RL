import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fractions import Fraction
import itertools as it

from gym import Env, spaces, ObservationWrapper
from gym.utils import seeding



# target: [y coord, target ratio]
# obstacles: [[[x_0,y_0,z_0], [x_1,y_1,z_0]], ...], an obstacle is a cuboid with spaned from [x_0,y_0,z_0] to [x_1,y_1,z_0]
# all dimensions in mm
def generate_random_map(rng):
    """ A map is a vector [y_min, y_max, y_o, i_n, i_d] i.e. max available vertical space,
        vertical coordinate of the output (input is always 0) and the target ratio written as a fraction
        i_n/i_d (starting ratio is always 1).
        All variables can take only integer values in milimiters
    """
    y_min, y_max = rng.randint(50, Y_MAX, size=2) # [incl, excl)
    y_min *= -1
    output = rng.randint(y_min+30,y_max-29)
    i_n, i_d = rng.randint(1, I_MAX, size=2)
    sign = [-1,1][np.random.random()<0.5]

    return [y_min, y_max, output, sign*i_n/i_d]

def get_admisible_ratios(i_max):
    """ First calculates admisible ratio for one gear stage given a number of steps
        (up and down) then calculates their n products to account for different combinations
    """
    # Ratios for traditional gear stage
    r = [-Fraction(*t) for t in it.product(np.arange(i_max)+1,np.arange(i_max)+1)]
    # # For planetery gear stage
    # for i in range(2,steps+1): # ring radius
    #     for j in range(1,i): # planete diamter
    #         r.append(Fraction(2*i-j,i-j))      
    # check for repetitions
    ratios=[]
    for frac in r:
        if frac not in ratios:
            ratios.append(float(frac))

    # stage_ratios=ratios[::]
    # for _ in range(n):
    #     for frac in [r1*r2 for r1,r2 in it.product(ratios,stage_ratios)]:
    #         if frac not in ratios:
    #             ratios.append(frac)
    return ratios

def cyl2car(x,r,phi):
    """ Transforms cylindrical coordinates to cartesian ones (x,y,z)"""
    return x, r*np.cos(phi), r*np.sin(phi)

# def get_cube(x_0,y_0,z_0,w,b,h):   
#     """ Creates a pointclout for a cuboid of dimensions W x B x H.
#         x_0,y_0,z_0 are the coordinates of the vertex closest to origin.
#     """
#     phi = np.arange(1,10,2)*np.pi/4
#     Phi, Theta = np.meshgrid(phi, phi)

#     x = w/2+x_0 + w*np.cos(Phi)*np.sin(Theta)
#     y = b/2+y_0 + b*np.sin(Phi)*np.sin(Theta)
#     z = h/2+z_0 + h*np.cos(Theta)/np.sqrt(2)
#     return x,y,z

# def generate_random_map(size=8, p=0.8):
#     valid = False

#     # DFS to check that it's a valid path.
#     def is_valid(res):
#         frontier, discovered = [], set()
#         frontier.append((0, 0))
#         while frontier:
#             r, c = frontier.pop()
#             if not (r, c) in discovered:
#                 discovered.add((r, c))
#                 directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
#                 for x, y in directions:
#                     r_new = r + x
#                     c_new = c + y
#                     if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
#                         continue
#                     if res[r_new][c_new] == 'G':
#                         return True
#                     if (res[r_new][c_new] != 'H'):
#                         frontier.append((r_new, c_new))
#         return False

#     while not valid:
#         p = min(1, p)
#         res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
#         res[0][0] = 'S'
#         res[-1][-1] = 'G'
#         valid = is_valid(res)
#     return ["".join(x) for x in res]


DIAMS = 5*np.arange(1,21) # admisible gear diamteres
PHI = np.linspace(0,2*np.pi,3)[:-1]
Y_MAX = 300 # max vertical gearbox size is restricted to [-Y_MAX, Y_MAX]
I_MAX = 5 # max target ratio will range from +-1/I_MAX to +-I_MAX
RATIOS = get_admisible_ratios(I_MAX)
PENALTY = 10 # fixed penalty if rotation direction is wrong

class GridSchematic3D(Env):
    def __init__(self, env_map=None, seed=None, static=False):
        """ - env_map is a vector [y_min, y_max, y_o, i_n, i_d],
            - seed may be used to have the pseudo random maps always appear 
            in the same order.
            - diams is a list of admisible gear diameters
            - static environments always use the same map
        """
        self.seed(seed)
        env_map = generate_random_map(self.np_random) if env_map is None else env_map
        self.i_target = env_map[-1]
        self.y_target = env_map[2] # target position
        self.y_lims = tuple(env_map[:2])
        self.static = static

        # state is a tuple (current_y, current_i, y_max, y_min, y_dist, i_dist)
        self.observation_space = spaces.Box(low=-Y_MAX, high=Y_MAX, shape=(6,), dtype=np.float32)
        # action is a tuple (primary_gear_diam, i, phi)
        self.action_space = spaces.Tuple((spaces.Discrete(len(DIAMS)), spaces.Discrete(len(RATIOS)), spaces.Discrete(len(PHI))))

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.reset()


    def step(self, a:tuple):
        """ Termination if:
            - gearbox output lies, near enough the target -> cost 1
            - more than 10 stages has been created -> cost 0
            Returns tuple: (new_state, cost, done, info)
            Every step is a new stage (that means stages can only consist of two gears)
            
            Unsolved: the only d1 that's valid if we remain in the same gear stage is the same
        """
        done = False
        self.stage_no += 1
        self.a_last = a
        d1_i, i_i, phi_i = a
        i = RATIOS[i_i]
        d1 = DIAMS[d1_i]
        phi = PHI[phi_i]
        d2 = d1/-i # that's where planetary graebox could be implemented
        self.y += ((d1+d2)/2)*np.cos(phi) 
        self.i *= i

        i_dist = abs(self.i_target - self.i)
        y_dist = self.y_target-self.y # signed distance
        self.s = new_state = (self.y, self.i, *self.y_lims, y_dist, i_dist)
        cost = 10*abs(i_dist) + abs(y_dist)

        if self.y+d2/2>self.y_lims[-1] or self.y-d2/2<self.y_lims[0]:
            done = True
            cost += 100
        if self.stage_no==10:
            done = True
            cost += 100
        if y_dist<1e-3 and i_dist<1e-3:
            done = True
            cost = PENALTY if self.i_target*self.i<0 else 0

        return new_state, cost, done


    def render(self, delay=0.1):
        """ Displays the state trajectory of the environment up to the
            current state. Works currently for 2D case only.
        """
        if self.a_last is not None: 
            d1_i, i_i, phi_i = self.a_last
            i = RATIOS[i_i]
            d1 = DIAMS[d1_i]
            phi = PHI[phi_i]
            y2_c = self.y
            d2 = d1/-i
            y1_c = y2_c - ((d1+d2)/2)*np.cos(phi)
            # plot shaft (just for readability)
            x_new = 10*self.stage_no + np.array([-10,0])
            self.ax.plot(x_new, y1_c*np.ones(2), np.zeros(2), c="tab:blue")
            # plot gears as circles
            n = 101
            k = np.linspace(0,1,n)
            x1, y1_r, z1 = cyl2car([x_new[-1]]*n, [d1/2]*n, 2*k*np.pi)
            self.ax.plot(x1, y1_r+[y1_c]*n, z1, c="tab:orange")
            x2, y2_r, z2 = cyl2car(x1, [d2/2]*n, 2*k*np.pi)
            self.ax.plot(x2, y2_r+[y2_c]*n, z2, c="tab:orange")
        # for obst in env.obstacles:
        #     w,b,h = [abs(x-y) for (x,y) in zip(*obst)]
        #     ax.plot_surface(*get_cube(*obst[0], w,b,h), color="tab:grey", alpha=0.3)
        plt.legend()
        plt.pause(delay)


    def reset(self):
        if not self.static: # generate new random map
            env_map = generate_random_map(self.np_random)
            self.i_target = env_map[-1]
            self.y_target = env_map[2] # target position
            self.y_lims = tuple(env_map[:2])
            
        self.y = 0
        self.i = 1.0
        self.stage_no = 0
        # state is a tuple (current_y, current_i, y_max, y_min, y_dist, i_dist)
        self.s = (self.y, self.i, *self.y_lims, self.y_target, abs(1 - self.i_target))
        self.a_last = None

        # PLot prep, plot I/O and y limits
        self.ax.clear()
        self.ax.scatter(0, self.y, 0, c="tab:green", label="Input")
        self.ax.scatter(100, self.y_target, 0, c="tab:red", label="Output")
        x,z = np.meshgrid(np.linspace(0,100,101), np.linspace(-Y_MAX,Y_MAX,101))
        y_l = self.y_lims[0] * np.ones_like(x)
        y_u = self.y_lims[-1] * np.ones_like(x)
        self.ax.plot_surface(x,y_l,z, color="tab:grey", alpha=0.3)
        self.ax.plot_surface(x,y_u,z, color="tab:grey", alpha=0.3)
        
        return self.s


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

class NormalizedOutput(ObservationWrapper):
    """ Should I normalize all geometric distances? like y_min y_max to 0,1
        current_y in [0,1] etc.
    """
    pass


if __name__=="__main__":
    env = GridSchematic3D()
    env.render()
    print(env.seed)
    print(f"Current y= {env.y}, distance to target= {env.y_target-env.y}")
    i_dist = abs(1.0 - env.i_target)
    print(f"Current ratio= {env.i}, distance to target= {i_dist}")
    # print(RATIOS)
        
    # play manually
    flag = True
    while flag:
        # wsad + distance e.g. 'w3' to go 3 steps up
        d,i,phi_i = input("Action (d1, ratio, up(0)/down(1)) or q to quit:").split()
        action = (int(d)//5, RATIOS.index(-float(i)), int(phi_i))

        new_state, cost, done = env.step(action)
        current_y, current_i, _, _, y_dist, i_dist = new_state
        print(f"Current y {current_y}, distance to target {y_dist}")
        print(f"Current ratio {current_i}, distance to target {i_dist}")
        env.render()
        if done:
            f = input("Start new env y/n?\n")
            if f=='y':
                env.reset()
            else:
                flag = False
            
        

    