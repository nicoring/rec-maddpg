import os
from collections import namedtuple, deque
import random
import pickle

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('observation', 'action', 'reward', 'next_observation', 'done'))
Batch = namedtuple('Batch', ('observations', 'actions', 'rewards', 'next_observations', 'dones'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    @staticmethod
    def sample_from_memories(memories, batch_size):
        """Collect experiences from all agents"""
        assert len(set(len(m) for m in memories)) == 1
        max_index = len(memories[0])
        index = np.random.randint(max_index, size=batch_size)
        batches = [m.sample_batch(batch_size, index) for m in memories]
        return Batch(*zip(*batches))

    def add(self, *args):
        """Saves a transition"""
        self.memory.append(Transition(*args))

    def sample_batch(self, batch_size, index=None):
        if index is None:
            transitions = random.sample(self.memory, batch_size)
        else:
            transitions = [self.memory[i] for i in index]
        batch = Batch(*zip(*transitions))
        observations = torch.tensor(batch.observations, dtype=torch.float)
        actions = torch.stack(batch.actions)
        rewards = torch.tensor(batch.rewards, dtype=torch.float).unsqueeze(1)
        next_observations = torch.tensor(batch.next_observations, dtype=torch.float)
        dones = torch.tensor(batch.dones).unsqueeze(1)
        return Batch(observations, actions, rewards, next_observations, dones)

    def __len__(self):
        return len(self.memory)

    def save(self, path):
        filename = os.path.join(path, 'memory.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, path):
        filename = os.path.join(path, 'memory.pkl')
        with open(filename, 'rb') as f:
            transitions = pickle.load(f)
            self.memory.extend(transitions)
