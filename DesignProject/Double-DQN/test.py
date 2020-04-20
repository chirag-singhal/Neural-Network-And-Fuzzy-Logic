import random
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import torch

from replaymemory import Transition, ReplayMemory
from input_extract import get_screen


# verbosity levels are 0, 1 or 2:-
# 0 - no additional info
# 1 - basic plots are visible
# 2 - plots and in-depth info visible
VERBOSITY = 2


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
env = gym.make('VideoPinball-v0')


def testPinballEnv():
	env.reset()
	if VERBOSITY == 1:
		print('Action space size = \t', env.action_space.n)
		print('Action space = \t', env.action_space)
		print('Observation space = \t', env.observation_space)
	env_rets = namedtuple('env_rets', (
		'observation',
		'reward',
		'done',
		'info'))
	env_ret = env_rets(*env.step(env.action_space.sample()))
	if VERBOSITY == 2:
		for key, item in env_ret._asdict().items():
			print(key ,'\t', type(item), '\t', item)
	print('VideoPinball-v0 - No errors')


def testReplayMemory():
	memory = ReplayMemory(1000)
	choices = list(range(9)) + [None]
	for i in range(500):
		state = np.arange(100).reshape(10, 10)
		action = random.choice(choices)
		next_state = random.choice(choices)
		reward = random.choice(choices)
		memory.push(state, action, next_state, reward)
	if VERBOSITY == 2:
		print(memory.sample(4))
		print(memory.sample(4, array_wise=True))
	print('ReplayMemory - No errors')


def testInputExtract():
	env.reset()
	# get_screen calls 'env.render()' internally
	screen = get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy()
	if VERBOSITY in [1, 2]:
		plt.figure()
		plt.imshow(screen, interpolation='none')
		plt.title('Example extracted screen')
		plt.show()
	print('InputExtract - No errors')


if __name__ == '__main__':
	print(device)
	testPinballEnv()
	testReplayMemory()
	testInputExtract()
