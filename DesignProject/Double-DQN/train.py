import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from replaymemory import Transition


def polyak_update(hyperparams, target_net, primary_net):	
	# theta' = Tau * theta + (1 - theta) * theta'
	for target_param, primary_param in zip(target_net.parameters(), primary_net.parameters()):
		target_param.data.copy_(hyperparams['TARGET_UPDATE_TAU']*primary_param.data + \
			target_param.data*(1.0 - hyperparams['TARGET_UPDATE_TAU']))


def optimize_model(primary_net, target_net, optimizer, memory, hyperparams, device):
	if len(memory) < hyperparams['BATCH_SIZE']:
		return
 
	batch = Transition(*zip(*memory.sample(hyperparams['BATCH_SIZE'])))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(
		lambda s: s is not None, batch.next_state)),
		device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of
	# actions taken. These are the actions which would've been taken for each batch
	# state according to primary_net
	state_action_values = primary_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# argmax of action for Q(s_{t+1}, a') by target network
	next_state_values = torch.zeros(hyperparams['BATCH_SIZE'], device=device)
	next_state_argvalues = target_net(non_final_next_states).max(1)[1].detach()
	# Use the Q(s_{t+1}, argmax_target) of primary network
	next_state_values[non_final_mask] = primary_net(non_final_next_states)[np.arange(next_state_argvalues.shape[-1]), next_state_argvalues]

	# Compute the expected Q values
	expected_state_action_values = (next_state_values * hyperparams['GAMMA']) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
	loss_temp = loss.item()

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in primary_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

	# Update target network parameters
	polyak_update(hyperparams, target_net, primary_net)
	return loss_temp
