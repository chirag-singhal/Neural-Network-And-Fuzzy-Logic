from collections import deque, namedtuple
import random

import numpy as np


Transition = namedtuple('transition', (
	'state',
	'action',
	'next_state',
	'reward'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.memory = deque(maxlen=capacity)

	def push(self, *args):
		""" Saves a transition """
		self.memory.append(Transition(*args))

	def sample(self, batch_size, array_wise=False):
		"""
		If 'array_wise' is 'False', a list of 'Transition' is returned.

		if 'array_wise' is 'True', then a 'Transition' with its members
		as numpy 1D arrays of the size of 'batch_size'
		"""
		samples = random.sample(self.memory, batch_size)
		if not array_wise:
			return samples
		samples = np.array(list(zip(*samples)))
		transition = Transition(
			samples[0],
			samples[1],
			samples[2],
			samples[3],
			)
		return transition


	def __len__(self):
		return len(self.memory)
