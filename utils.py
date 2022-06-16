import numpy as np
import pickle
import gzip
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
	def __init__(self):
		self.storage = dict()
		self.buffer_size = 1000000
		self.ctr = 0

	def add(self, data):
		self.storage['observations'][self.ctr] = data[0]
		self.storage['next_observations'][self.ctr] = data[1]
		self.storage['actions'][self.ctr] = data[2]
		self.storage['rewards'][self.ctr] = data[3]
		self.storage['terminals'][self.ctr] = data[4]
		self.ctr += 1
		self.ctr = self.ctr % self.buffer_size

	def sample(self, batch_size):
		ind = np.random.randint(0, self.buffer_size, size=batch_size)
		obs = self.storage['observations'][ind]
		ac = self.storage['actions'][ind]
		rew = self.storage['rewards'][ind]
		next_obs = self.storage['next_observations'][ind]
		done = self.storage['terminals'][ind]
		return (np.array(obs),
				np.array(next_obs),
				np.array(ac),
				np.array(rew).reshape(-1, 1),
				np.array(done).reshape(-1, 1))

	def normalize_states(self, eps=1e-3):
		mean = self.storage['observations'].mean(0, keepdims=True)
		std = self.storage['observations'].std(0, keepdims=True) + eps
		self.storage['observations'] = (self.storage['observations'] - mean)/std
		self.storage['next_observations'] = (self.storage['next_observations'] - mean)/std
		return mean, std

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename, bootstrap_dim=None):
		"""Deprecated, use load_hdf5 in main.py with the D4RL environments""" 
		with gzip.open(filename, 'rb') as f:
				self.storage = pickle.load(f)
		
		sum_returns = self.storage['rewards'].sum()
		num_traj = self.storage['terminals'].sum()
		if num_traj == 0:
				num_traj = 1000
		average_per_traj_return = sum_returns/num_traj
		print ("Average Return: ", average_per_traj_return)
		# import ipdb; ipdb.set_trace()
		
		num_samples = self.storage['observations'].shape[0]
		if bootstrap_dim is not None:
				self.bootstrap_dim = bootstrap_dim
				bootstrap_mask = np.random.binomial(n=1, size=(1, num_samples, bootstrap_dim,), p=0.8)
				bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
				self.storage['bootstrap_mask'] = bootstrap_mask[:num_samples]
