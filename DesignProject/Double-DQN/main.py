import math
import random
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import torch
import gym

from dqn import DQN
from input_extract import get_screen
from replaymemory import ReplayMemory
from train import polyak_update, optimize_model


MODEL_SAVE_PATH = None
SHOW_PLOTS = True


if MODEL_SAVE_PATH is None:
	import os
	MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'models/model_0.pth')


def matplotlib_sanity_check(env, device):
	env.reset()
	plt.figure()
	plt.imshow(get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy()
		, interpolation='none')
	plt.title('Example extracted screen')
	plt.show()


def plot_durations(episode_durations):
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episodes averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())


def select_action(network, state, n_actions, steps_done, hyperparams, device):
	# state is a single state, not a batch

	sample = random.random()

	# 'EPS_THRESHOLD' is the threshold for Exploration vs. Exploitation
	# As 'steps_done' goes from 0 to some N, 'eps_threshold' goes from 'EPS_START'
	# to 'EPS_END', in exponential fashion with time constant as 'EPS_DECAY'
	# EPS_THRES = b + (a - b)*e^(-t/T)

	eps_threshold = hyperparams['EPS_END'] + (hyperparams['EPS_START'] - \
		hyperparams['EPS_END']) * math.exp(-1. * steps_done / hyperparams['EPS_DECAY'])
	
	if sample > eps_threshold:
		with torch.no_grad():
			return network(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def main():
	env = gym.make('VideoPinball-v0')
	device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

	hyperparameters = {
		'BATCH_SIZE' : 64,
		'GAMMA' : 0.999,
		'EPS_START' : 0.9,
		'EPS_END' : 0.05,
		'EPS_DECAY' : 200,
		'TARGET_UPDATE_TAU': 0.01
		}

	if SHOW_PLOTS:
		matplotlib_sanity_check(env, device)

	## ENVIRONMENT
	env.reset()
	# 'env.render()' just after 'env.reset()' returns same observation for both
	init_screen = get_screen(env, device)
	# 'get_screen()' returns shape in (BCHW)
	_, _, screen_height, screen_width = init_screen.shape

	n_actions = env.action_space.n
	##

	## NETWORKS
	primary_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net.load_state_dict(primary_net.state_dict())
	##


	optimizer = optim.RMSprop(primary_net.parameters())
	memory = ReplayMemory(10000)

	steps_done = 0
	episode_durations = []

	# Main training loop
	num_episodes = 50
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		env.reset()
		last_screen = get_screen(env, device)
		current_screen = get_screen(env, device)
		state = current_screen - last_screen
		for t in count():
			# Select and perform an action
			action = select_action(primary_net, state, n_actions, steps_done, hyperparameters, device)
			steps_done += 1
			_, reward, done, _ = env.step(action.item())
			reward = torch.tensor([reward], device=device)

			# Observe new state
			last_screen = current_screen
			current_screen = get_screen(env, device)
			if not done:
				next_state = current_screen - last_screen
			else:
				next_state = None

			# Store the transition in memory
			memory.push(state, action, next_state, reward)

			# Move to the next state
			state = next_state

			# Perform one step of the optimization (on the primary network)
			loss_temp = optimize_model(primary_net, target_net, optimizer, memory, hyperparameters, device)
			if done:
				episode_durations.append(t+1)
				if SHOW_PLOTS:
					plot_durations(episode_durations)
				break
		# # Update the target network, copying all weights and biases in DQN
		# if i_episode % hyperparameters['TARGET_UPDATE'] == 0:
		# 	target_net.load_state_dict(primary_net.state_dict())
		print('Episode %d :- Loss = %f' % ((i_episode+1), loss_temp))

	print('\nTraining Done!')

	## Save model
	torch.save(primary_net.state_dict(), MODEL_SAVE_PATH)
	##

	env.render()
	env.close()


if __name__ == '__main__':	

	# set up matplotlib
	is_ipython = 'inline' in matplotlib.get_backend()
	if is_ipython:
		from IPython import display

	plt.ion()

	main()

	plt.ioff()
	plt.show()
