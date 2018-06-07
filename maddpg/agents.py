import numpy as np
import torch

from memory import ReplayBuffer


class RandomAgent:
    def __init__(self, index, name, env):
        self.env = env
        self.index = index
        self.name = name
        self.num_actions = self.env.action_space[self.index].n
    
    def act(self, obs):
        logits = np.random.sample(self.num_actions)
        return logits / np.sum(logits)
    
    def experience(self, obs, action, reward, new_obs, done):
        pass

    def update(self, agents):
        pass

class MaddpgAgent:
    def __init__(self, index, name, actor, critic, params):
        self.index = index
        self.name = name
        self.actor = actor
        self.critic = critic
        self.actor_target = actor.clone()
        self.critic_target = critic.clone()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-2)
        self.memory = ReplayBuffer(params.memory_size)
        self.batch_size = params.batch_size
        self.tau = params.tau
        self.gamma = params.gamma

    def update_params(self, target, source):
        zipped = zip(target.parameters(), source.parameters())
        for target_param, source_param in zipped:
            updated_param = target_param.data * (1 - self.tau) + \
                source_param.data * self.tau
            target_param.data.copy_(updated_param)

    def act(self, obs):
        return self.actor.act_rand(torch.tensor(obs, dtype=torch.float))
    
    def experience(self, obs, action, reward, new_obs, done):
        self.memory.add(obs, action, reward, new_obs, float(done))

    def train_actor(self, batch):
        # forward pass
        pred_actions = self.actor.act_det(batch.observations[self.index])
        actions = list(batch.actions)
        actions[self.index] = pred_actions
        pred_q = self.critic(batch.observations, actions)
        # backward pass        
        loss = -pred_q.mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss

    @staticmethod
    def mse(a, b):
        return torch.mean((a - b)**2)

    def train_critic(self, batch, agents):
        """Train critic with q-learning loss."""
        # forward pass
        # (a_1', ..., a_n') = (mu'_1(o_1'), ..., mu'_n(o_n'))
        pred_actions = [a.actor_target.act_det(o)
                        for o, a in zip(batch.next_observations, agents)]

        # if not done: y = r + gamma * Q(o_1, ..., o_n, a_1', ..., a_n')  
        # if done:     y = r
        target_q = batch.rewards[self.index] + (1.0 - batch.dones[self.index]) * \
                                               self.gamma * \
                                               self.critic_target(batch.next_observations, pred_actions)

        # backward pass
        # loss(params) = mse(y, Q(o_1, ..., o_n, a_1, ..., a_n))
        loss = self.mse(target_q, self.critic(batch.observations, batch.actions))

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss

    def update(self, agents):
        # sample minibatch
        memories = [a.memory for a in agents]
        batch = ReplayBuffer.sample_from_memories(memories, self.batch_size)
        
        # train networks
        actor_loss = self.train_actor(batch)
        critic_loss = self.train_critic(batch, agents)

        # update target network params
        self.update_params(self.actor_target, self.actor)
        self.update_params(self.critic_target, self.critic)

        return actor_loss, critic_loss
