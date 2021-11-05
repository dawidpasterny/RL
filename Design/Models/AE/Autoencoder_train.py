import sys
import os
sys.path.append(os.getcwd())

import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque

from Design.Models.AE import model as ae
import Design.Environments.stage_creator as sc
import torch.multiprocessing as mp

LEARNING_RATE = 1e-4
LOCAL_BATCH_SIZE = 84
NUM_THREADS = 9
BATCH_SIZE = LOCAL_BATCH_SIZE*NUM_THREADS


def random_exp_gen(env, batch_size):
    """ Using normal distribution """
    done=False
    env.reset()
    while True:
        batch = torch.zeros(batch_size,1,env.N,env.N)
        for i in range(batch_size):
            j=0
            while j<10:
                d = np.clip(np.random.normal(.35,.25), sc.D_MIN, sc.D_MAX)
                phi = np.random.rand()
                obs,_,done,_ = env.step([d,phi]+[1])
                screen = obs["observation"][0]
                # env.render(delay=.1)
                j += 1
            batch[i] = torch.FloatTensor([screen]) # add screen
            env.reset()
            done=False

        yield batch


def kernel(train_queue, device):
    env = sc.StageCreator(boundary=.5, mode="selfplay")
    env = sc.ScreenOutput(84, env)
    exp_gen = random_exp_gen(env, LOCAL_BATCH_SIZE)

    for batch in exp_gen:
        train_queue.put(batch.to(device))


device = "cuda" if torch.cuda.is_available() else "cpu"
auten = ae.Autoencoder84(1,False).to(device)
print(auten)
optimizer = torch.optim.Adam(auten.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
writer = SummaryWriter(log_dir="./Design/Models/AE/autoencoder-runs/"+datetime.datetime.now().strftime("%b%d_%H_%M_%S"))
best_loss = None


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    train_queue = mp.Queue(maxsize=NUM_THREADS)
    proc_list = []
    for i in range(NUM_THREADS):
        p = mp.Process(target=kernel, args=(train_queue, device))
        p.start()
        proc_list.append(p)

    try:
        training_batch = deque(maxlen=NUM_THREADS)
        while True:
            try:
                training_batch.append(train_queue.get(block=False))
            except:
                # print("No new batch found")
                pass

            if len(training_batch)<NUM_THREADS:
                continue

            batch = torch.cat(tuple(training_batch)).to(device)
            optimizer.zero_grad()
            out = auten.forward(batch)
            loss = nn.BCELoss()(out, batch)
            loss.backward()
            optimizer.step()

            if best_loss is None or loss < best_loss:
                torch.save(auten.state_dict(), "./Design/Models/AE/Autoencoder84_1.dat")
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

    finally:
        for p in proc_list:
            p.terminate()
            p.join()
