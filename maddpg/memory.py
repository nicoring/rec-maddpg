import os
from collections import namedtuple
import random
import pickle

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('observation', 'action', 'reward', 'next_observation', 'done'))
Batch = namedtuple('Batch', ('observations', 'actions', 'rewards', 'next_observations', 'dones'))


class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.next_idx = 0

    def sample_index(self, batch_size, max_past=None):
        if max_past:
            min_idx = self.next_idx - max_past
            max_idx = self.next_idx
            if not self.full():
                min_idx = min_idx if min_idx >= 0 else 0
            index = np.random.randint(min_idx, max_idx, size=batch_size)
            if self.full():
                index = [i % self.capacity for i in index]
            return index
        else:
            min_idx = 0
            max_idx = len(self)
            return np.random.randint(min_idx, max_idx, size=batch_size)

    def append(self, data):
        if not self.full():
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.capacity

    def full(self):
        return not self.next_idx >= len(self.storage)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, key):
        return self.storage[key]

    def __setitem__(self, key, value):
        self.storage[key] = value

    def __iter__(self):
        return iter(self.storage)

    def __repr__(self):
        return self.storage.__repr__()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = RingBuffer(capacity)

    @staticmethod
    def sample_from_memories(memories, batch_size, max_past=None):
        """Collect experiences from all agents"""
        assert len(set(m.memory.next_idx for m in memories)) == 1
        index = memories[0].memory.sample_index(batch_size, max_past)
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
        observations = torch.tensor(batch.observations, dtype=torch.float).to(device)
        actions = torch.stack(batch.actions).to(device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float).unsqueeze(1).to(device)
        next_observations = torch.tensor(batch.next_observations, dtype=torch.float).to(device)
        dones = torch.tensor(batch.dones).unsqueeze(1).to(device)
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
            self.memory = pickle.load(f)

if __name__ == '__main__':
    print('run tests')
    b = RingBuffer(10)
    b.append(1)
    b.append(2)
    idx = b.sample_index(100)
    assert all(i == 0 or i == 1 for i in idx)
    idx = b.sample_index(100, max_past=5)
    assert all(i == 0 or i == 1 for i in idx)
    b.append(3)
    b.append(4)
    b.append(5)
    b.append(6)
    b.append(7)
    b.append(8)
    b.append(9)
    b.append(10)
    idx = b.sample_index(1000, max_past=5)
    valid = [5, 6, 7, 8, 9]
    assert all(i in valid for i in idx)
    b.append(11)
    b.append(12)
    assert b.next_idx == 2, 'next_idx should be 2 is %d' % b.next_idx
    idx = b.sample_index(1000, max_past=5)
    valid = [7, 8, 9, 0, 1]
    assert all(i in valid for i in idx)
    print('all tests successful')
