import sys
import os
sys.path.append(os.getcwd())

from lib import model
import datetime
from Design.Environments import stage_creator as sc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque


LEARNING_RATE = 1e-3
BATCH_SIZE = 64

env = sc.StageCreator(seed=3672871121734420758, boundary=False)
env = sc.ScreenOutput(64, env)

def random_exp_gen(env, batch_size):
    """ Using normal distribution """
    done=False
    env.reset()
    while True:
        batch = torch.zeros(batch_size,1,env.N,env.N)
        for i in range(batch_size):
            while not done:
                d = np.clip(np.random.normal(.35,.25), sc.D_MIN, sc.D_MAX)
                phi = np.random.rand()
                next_state,_,done,_ = env.step((d,phi))
            batch[i] = torch.tensor([next_state[0]])
            env.reset()
            done=False

        yield batch

auten = model.Autoencoder(1,False)
optimizer = torch.optim.Adam(auten.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
loss_crt = nn.MSELoss()
writer = SummaryWriter(log_dir="./Design/Models/DDPG/autoencoder-runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))
best_loss = None
exp_gen = random_exp_gen(env,64)

for i, batch in enumerate(exp_gen):
    optimizer.zero_grad()
    out = auten.forward(batch)
    loss = loss_crt(out, batch)
    loss.backward()
    optimizer.step()

    if best_loss is None or loss < best_loss:
        torch.save(auten.state_dict(), "./Design/Models/DDPG/Autoencoder-best.dat")
        if best_loss is not None:
            print(f"Best loss updated {best_loss} -> {loss}, model saved")
        best_loss = loss

    writer.add_scalar("loss", loss, i)
    # writer.add_scalar("speed", speed, frame_idx)

    print(f"batch: {i}, loss: {loss}")


# # Visual check
# batch = next(random_exp_gen(env,10)).detach().numpy()

# fig, axs = plt.subplots(2, 5)
# for i,ax in enumerate(axs.flat):
#     ax.pcolormesh(batch[i][0], cmap="binary")
# plt.show()
