from collections import OrderedDict, namedtuple

import torch
import numpy as np

Batch = namedtuple('Batch', ('observations', 'actions', 'rewards', 'next_observations', 'dones'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RingBuffer:
    def __init__(self, capacity, max_epsiode_len, data_dim):
        self.capacity = capacity
        self.max_epsiode_len = max_epsiode_len
        self.storage = np.zeros((capacity, max_epsiode_len, data_dim))
        self.episode_lengths = np.zeros(capacity, dtype=np.int)
        self.last_episode = 0
        self.last_index = 0

    def __len__(self):
        if not self.full():
            return self.last_episode + 1
        else:
            return self.capacity

    def full(self):
        return self.last_episode + 1 >= self.capacity

    def store(self, episode_num, data):
        current_index = episode_num % self.capacity
        if episode_num > self.last_episode:
            self.episode_lengths[current_index] = 0
        timestep = self.episode_lengths[current_index]
        self.storage[current_index, timestep, :] = data
        self.episode_lengths[current_index] += 1
        self.last_episode = episode_num
        self.last_index = current_index

    def sample_episode_index(self, batch_size, max_past=None):
        if max_past:
            max_idx = self.last_index
            min_idx = max_idx - max_past
            if not self.full():
                min_idx = min_idx if min_idx >= 0 else 0
            index = np.random.randint(min_idx, max_idx + 1, size=batch_size)
            if self.full():
                index = index % self.capacity
            return index
        else:
            min_idx = 0
            max_idx = len(self)
            return np.random.randint(min_idx, max_idx, size=batch_size)

    def sample_transition_index(self, batch_size, max_past=None):
        episode_nums = self.sample_episode_index(batch_size, max_past)
        lengths = self.episode_lengths[episode_nums]
        timesteps = np.random.randint(25, size=batch_size) % lengths
        return episode_nums * self.max_epsiode_len + timesteps

    def sample_episodes(self, index):
        return self.storage[index, :, :], self.episode_lengths[index]

    def sample_transitions(self, index):
        return self.storage.reshape(self.capacity * self.max_epsiode_len, -1)[index, :]


class ReplayMemory:
    def __init__(self, capacity, max_epsiode_len, n_actions, n_obs):
        self.capacity = capacity
        self.buffers = OrderedDict()
        self.buffers['observations'] = RingBuffer(capacity, max_epsiode_len, n_obs)
        self.buffers['actions'] = RingBuffer(capacity, max_epsiode_len, n_actions)
        self.buffers['rewards'] = RingBuffer(capacity, max_epsiode_len, 1)
        self.buffers['next_observations'] = RingBuffer(capacity, max_epsiode_len, n_obs)
        self.buffers['dones'] = RingBuffer(capacity, max_epsiode_len, 1)

    @staticmethod
    def sample_episodes_from(memories, batch_size, max_past=None):
        """Samples episodes from all agents"""
        assert len(set(b.last_episode for m in memories for b in m.buffers.values())) == 1
        index = memories[0].buffers['actions'].sample_episode_index(batch_size, max_past)
        batches, lengths = list(zip(*(m.create_episode_batch(index) for m in memories)))
        return Batch(*(list(zipped) for zipped in zip(*batches))), lengths[0]

    @staticmethod
    def sample_transitions_from(memories, batch_size, max_past=None):
        """Samples transitions from all agents"""
        assert len(set(b.last_episode for m in memories for b in m.buffers.values())) == 1
        index = memories[0].buffers['actions'].sample_transition_index(batch_size, max_past)
        batches = (m.create_transition_batch(index) for m in memories)
        return Batch(*(list(zipped) for zipped in zip(*batches)))

    def add(self, episode_num, obs, action, reward, next_obs, done):
        """Saves a transition"""
        for buffer, data in zip(self.buffers.values(), [obs, action, reward, next_obs, done]):
            buffer.store(episode_num, data)

    @staticmethod
    def create_tensor(data, ensure_dim):
        """Creates float tensor and moves it to the GPU if available"""
        tensor = torch.tensor(data, requires_grad=False,
                              dtype=torch.float, device=DEVICE)
        if len(tensor.size()) < ensure_dim:
            tensor = tensor.unsqueeze(-1)
        return tensor

    def create_episode_batch(self, index):
        """Creates minibatch of episodes, which are zero-padded to be of equal length"""
        data = [b.sample_episodes(index) for b in self.buffers.values()]
        data, lengths = zip(*data)
        data = [self.create_tensor(a, 3).transpose(0, 1) for a in data]
        return Batch(*data), lengths[0]

    def create_transition_batch(self, index):
        """Creates minibatch of transitions"""
        data = [self.create_tensor(b.sample_transitions(index), 2) for b in self.buffers.values()]
        return Batch(*data)
