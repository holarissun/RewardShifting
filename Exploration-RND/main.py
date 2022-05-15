import numpy as np
import gym
from dqn_rnd import DQN_RND
import matplotlib.pyplot as plt
from smooth_signal import smooth
import torch
import argparse
from gym_minigrid.wrappers import *


parser = argparse.ArgumentParser()
parser.add_argument("--max_epoch", default=1000, type=int)   # Max time steps to run environment or train for (this defines buffer size)
parser.add_argument("--gamma", default=0.95,type=float)                 # Discount factor
parser.add_argument("--gpu", default = 0, type = int)
parser.add_argument("--mode",default = 'RND', type=str)
parser.add_argument("--buffer_size",default=10000,type=int)
parser.add_argument("--repeat",default = 0, type = int)
parser.add_argument("--mn",default = 0, type = float)
parser.add_argument("--max_game_len",default = 200, type=int)
parser.add_argument("--env_name",default='MountainCar-v0', type=str)

args = parser.parse_args()

torch.cuda.set_device(args.gpu)


env = gym.make(args.env_name)

gamma = args.gamma
alg = DQN_RND(env,gamma,args)


num_epochs = args.max_epoch
for i in range(num_epochs):
    log = alg.run_epoch()
    print('epoch: {}. return: {}'.format(i,np.round(log.get_current('real_return')),2))
    print('mean intrinsic reward', np.mean(alg.intrinsic_reward_history[-200:]))
    Y = np.asarray(log.get_log('real_return'))
    np.save(f'npys/{args.env_name}_{args.gamma}_{args.buffer_size}_{args.max_epoch}_{args.mode}_{args.mn}_{args.repeat}.npy',Y)
    np.save(f'npys/{args.env_name}_{args.gamma}_{args.buffer_size}_{args.max_epoch}_{args.mode}_{args.mn}_{args.repeat}_intrinsic.npy',
            alg.intrinsic_reward_history)

Y = np.asarray(log.get_log('real_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(x, Y, Y2)

Y = np.asarray(log.get_log('combined_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
