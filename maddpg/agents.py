import itertools as it

import numpy as np
import torch

from memory import ReplayMemory
from models import Actor, LSTMActor
import distributions
from torch.distributions import RelaxedOneHotCategorical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    def act(self, obs, **kwargs):
        raise NotImplementedError

    def experience(self, episode_count, obs, action, reward, new_obs, done):
        pass

    def update(self, agents):
        pass


class SpreadScriptedAgent(Agent):
    def __init__(self, index, name, env):
        self.env = env
        self.index = index
        self.name = name

    @staticmethod
    def length(a):
        return np.sqrt(np.sum(a**2))

    @staticmethod
    def acc2action(acc):
        action = np.zeros(5)
        for i, a in enumerate(acc):
            if a >= 0:
                action[1+i*2] = a
            else:
                action[2+i*2] = -a
        if abs(np.sum(action)) > 0:
            action = action / np.sum(action)
        return action

    def get_target(self, agents, landmarks):
        matchings = [list(zip(agents, p)) for p in it.permutations(landmarks)]
        dists = [sum(self.length(l - a) for a, l in m) for m in matchings]
        best_matching = matchings[np.argmin(dists)]
        return best_matching[self.index][1]

    def act(self, obs, **kwargs):
        # vel = obs[:2]
        l1 = obs[2:4]
        l2 = obs[4:6]
        l3 = obs[6:8]
        a1 = obs[8:10]
        a2 = obs[10:12]
        # target = self.get_target([l1, l2, l3], [a1, a2])
        landmarks = [l1, l2, l3]
        agents = [a1, a2]
        agents.insert(self.index, [0, 0])
        target = self.get_target(agents, landmarks)
        return self.acc2action(target)


class RandomAgent(Agent):
    def __init__(self, index, name, env):
        self.env = env
        self.index = index
        self.name = name
        self.num_actions = self.env.action_space[self.index].n

    def act(self, obs, **kwargs):
        logits = np.random.sample(self.num_actions)
        return logits / np.sum(logits)


class MaddpgAgent(Agent):
    def __init__(self, index, name, env, actor, critic, params):
        self.index = index
        self.name = name
        self.env = env

        self.actor = actor.to(DEVICE)
        self.critic = critic.to(DEVICE)
        self.actor_target = actor.clone().to(DEVICE)
        self.critic_target = critic.clone().to(DEVICE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=params.lr_critic)
        self.memory = ReplayMemory(params.memory_size, params.max_episode_len,
                                   self.actor.n_outputs, self.actor.n_inputs)
        self.mse = torch.nn.MSELoss()

        # params
        self.batch_size = params.batch_size
        self.tau = params.tau
        self.gamma = params.gamma
        self.clip_grads = True

        # flags
        # local obs/actions means only the obs/actions of this agent are available
        # if obs and actions are local this is equivalent to DDPG
        self.local_obs = params.local_obs
        self.local_actions = params.local_actions

        # agent modeling
        self.use_agent_models = params.use_agent_models
        self.agent_models = {}
        self.model_optims = {}
        self.model_lr = params.modeling_lr
        self.entropy_weight = 1e-3
        self.max_past = params.max_past
        self.modeling_train_steps = params.modeling_train_steps
        self.modeling_batch_size = params.modeling_batch_size
        self.model_class = Actor

        # action and observation noise
        self.obfuscate_others = (params.sigma_noise is not None) or (params.temp_noise is not None)
        self.sigma_noise = params.sigma_noise
        self.temp_noise = params.temp_noise

    def init_agent_models(self, agents):
        for agent in agents:
            if agent is self:
                continue
            agent_model = self.model_class.from_actor(agent.actor).to(DEVICE)
            self.agent_models[agent.index] = agent_model
            optim = torch.optim.Adam(agent_model.parameters(), lr=self.model_lr)
            self.model_optims[agent.index] = optim

    def update_params(self, target, source):
        zipped = zip(target.parameters(), source.parameters())
        for target_param, source_param in zipped:
            updated_param = target_param.data * (1.0 - self.tau) + \
                source_param.data * self.tau
            target_param.data.copy_(updated_param)

    def act(self, obs, explore=True):
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(DEVICE)
        actions = self.actor.select_action(obs, explore=explore).detach()
        return actions.to('cpu').numpy()

    def experience(self, episode_count, obs, action, reward, new_obs, done):
        self.memory.add(episode_count, obs, action, reward, new_obs, float(done))

    def train_actor(self, batch):
        ### forward pass ###
        pred_actions = self.actor.select_action(batch.observations[self.index])
        actions = list(batch.actions)
        actions[self.index] = pred_actions
        q_obs = [batch.observations[self.index]] if self.local_obs else batch.observations
        q_actions = [actions[self.index]] if self.local_actions else actions
        pred_q = self.critic(q_obs, q_actions)
        ### backward pass ###
        p_reg = torch.mean(self.actor.forward(batch.observations[self.index])**2)
        loss = -pred_q.mean() + 1e-3 * p_reg
        self.actor_optim.zero_grad()
        loss.backward()
        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        return loss

    def train_critic(self, batch, agents):
        """Train critic with TD-target."""
        ### forward pass ###
        # (a_1', ..., a_n') = (mu'_1(o_1'), ..., mu'_n(o_n'))
        self_obs = batch.next_observations[self.index]
        self_action = self.actor_target.select_action(self_obs).detach()
        if self.local_actions or self.local_obs:
            q_next_actions = [self_action]
        else:
            if self.use_agent_models:
                q_next_actions = [m.select_action(batch.next_observations[idx]).detach()
                                  for idx, m in self.agent_models.items()]
                q_next_actions.insert(self.index, self_action)
            else:
                q_next_actions = [a.actor_target.select_action(o).detach()
                                  for o, a in zip(batch.next_observations, agents)]

        q_next_obs = [batch.next_observations[self.index]] if self.local_obs else batch.next_observations
        q_next = self.critic_target(q_next_obs, q_next_actions)
        reward = batch.rewards[self.index]
        done = batch.dones[self.index]

        # if not done: y = r + gamma * Q(o_1, ..., o_n, a_1', ..., a_n')
        # if done:     y = r
        q_target = reward + (1.0 - done) * self.gamma * q_next

        ### backward pass ###
        # loss(params) = mse(y, Q(o_1, ..., o_n, a_1, ..., a_n))
        q_obs = [batch.observations[self.index]] if self.local_obs else batch.observations
        q_actions = [batch.actions[self.index]] if (self.local_actions or self.local_obs) else batch.actions
        loss = self.mse(self.critic(q_obs, q_actions), q_target.detach())

        self.critic_optim.zero_grad()
        loss.backward()
        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        return loss

    def train_models(self, batch, agents):
        for idx, model in self.agent_models.items():
            obs = batch.observations[idx]
            actions = batch.actions[idx]
            distributions = model.prob_dists(obs)
            split_actions = torch.split(actions, agents[idx].actor.action_split, dim=-1)
            self.model_optims[idx].zero_grad()
            losses = torch.zeros(len(distributions))
            for i, (actions, dist) in enumerate(zip(split_actions, distributions)):
                entropy = dist.base_dist._categorical.entropy()
                loss = (dist.log_prob(actions).mean() + self.entropy_weight * entropy).mean()
                losses[i] = loss
            loss = -torch.mean(losses)
            loss.backward()
            self.model_optims[idx].step()
            return loss

    def compare_models(self, agents, batch):
        kls = []
        for idx, model in self.agent_models.items():
            kls.append([])
            obs = batch.observations[idx]
            modelled_distributions = model.prob_dists(obs)
            agent_distributions = agents[idx].actor.prob_dists(obs)
            for model_dist, agent_dist in zip(modelled_distributions, agent_distributions):
                kl_div = torch.distributions.kl.kl_divergence(agent_dist, model_dist).data
                kls[-1].append(kl_div.mean())
        return zip(self.agent_models.keys(), kls)

    def add_noise_(self, batch):
        for i in range(len(batch.actions)):
            if i == self.index:
                continue
            # get observations and actions for agent i
            obs = batch.observations[i]
            actions = batch.actions[i]
            # create noise tensors, same shape and on same device
            if self.sigma_noise is not None:
                obs = obs + torch.randn_like(obs) * self.sigma_noise
            if self.temp_noise is not None:
                temp = torch.tensor(self.temp_noise, dtype=torch.float, device=actions.device)
                actions = RelaxedOneHotCategorical(temp, probs=actions).sample()
            # add noise
            batch.observations[i] = obs
            batch.actions[i] = actions

    def update(self, agents):
        # collect transistion memories form all agents
        memories = [a.memory for a in agents]

        # train model networks
        if self.use_agent_models:
            model_losses = []
            for _ in range(self.modeling_train_steps):
                batch = self.memory.sample_transitions_from(memories,
                                                            self.modeling_batch_size,
                                                            max_past=self.max_past)
                if self.obfuscate_others:
                    self.add_noise_(batch)
                model_losses.append(self.train_models(batch, agents).data)
            model_loss = np.mean(model_losses)
            model_kls = self.compare_models(agents, batch)
        else:
            model_loss = None
            model_kls = None

        # sample minibatch
        batch = self.memory.sample_transitions_from(memories, self.batch_size)
        if self.obfuscate_others:
            self.add_noise_(batch)
        # train actor and critic network
        actor_loss = self.train_actor(batch)
        critic_loss = self.train_critic(batch, agents)

        # update target network params
        self.update_params(self.actor_target, self.actor)
        self.update_params(self.critic_target, self.critic)

        return actor_loss, critic_loss, model_loss, model_kls

    def get_state(self):
        if self.agent_models:
            models = {i: m.state_dict() for i, m in self.agent_models.items()}
            optims = {i: o.state_dict() for i, o in self.model_optims.items()}
            model_pair = (models, optims)
        else:
            model_pair = None
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, model_pair

    def load_state(self, state):
        for key, value in state['state_dicts'].items():
            getattr(self, key).load_state_dict(value)
        if 'models' in state:
            models, optims = state['models']
            for i, m in models.items():
                self.agent_models[i].load_state_dict(m)
            for i, o in optims.items():
                self.model_optims[i].load_state_dict(o)


class MARDPGAgent(MaddpgAgent):

    def __init__(self, index, name, env, actor, critic, params):
        super().__init__(index, name, env, actor, critic, params)
        self.max_episode_len = params.max_episode_len

    def mask(self, tensor, lengths):
        # tensor: T x m x d
        # lengths: m x 1 --> length of every element in the minibatch
        binary_mask = torch.ones_like(tensor)
        for i, length in enumerate(lengths):
            if length < 25:
                binary_mask[length:, i, :] = 0.0
        return tensor * binary_mask

    def add_noise_(self, batch, lengths=None):
        super().add_noise_(batch)
        if lengths is not None:
            for i in range(len(batch.actions)):
                if i == self.index:
                    continue
                if self.sigma_noise is not None:
                    obs = batch.observations[i]
                    batch.observations[i] = self.mask(obs, lengths)
                if self.temp_noise is not None:
                    actions = batch.actions[i]
                    batch.actions[i] = self.mask(actions, lengths)

    def train_actor(self, batch, lengths):
        ### forward pass ###
        pred_actions = self.actor.select_action(batch.observations[self.index])
        actions = list(batch.actions)
        actions[self.index] = pred_actions
        # TODO: recurrent
        pred_q = self.critic(batch.observations, actions, detached_states=True)
        pred_q = self.mask(pred_q, lengths)

        ### backward pass ###
        p_reg = torch.mean(self.actor.forward(batch.observations[self.index])**2)
        loss = -pred_q.mean() + 1e-3 * p_reg
        self.actor_optim.zero_grad()
        loss.backward()
        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        return loss

    def train_critic(self, batch, agents, lengths):
        """Train critic with TD-target."""
        ### forward pass ###
        # (a_1', ..., a_n') = (mu'_1(o_1'), ..., mu'_n(o_n'))
        if self.use_agent_models:
            pred_next_actions = [m.select_action(batch.next_observations[idx]).detach()
                                 for idx, m in self.agent_models.items()]
            self_obs = batch.next_observations[self.index]
            self_action = self.actor_target.select_action(self_obs).detach()
            pred_next_actions.insert(self.index, self_action)
        else:
            pred_next_actions = [a.actor_target.select_action(o).detach()
                                 for o, a in zip(batch.next_observations, agents)]

        # TODO: recurrent
        # out shape: timesteps x batch_size x 1
        q_next = self.critic_target(batch.next_observations, pred_next_actions)
        reward = batch.rewards[self.index]
        done = batch.dones[self.index]

        # if not done: y = r + gamma * Q(o_1, ..., o_n, a_1', ..., a_n')
        # if done:     y = r
        q_target = reward + (1.0 - done) * self.gamma * q_next

        ### backward pass ###
        # loss(params) = mse(y, Q(o_1, ..., o_n, a_1, ..., a_n))
        q_obs = batch.observations
        q_actions = batch.actions
        # TODO: recurrent
        q_pred = self.critic(q_obs, q_actions)
        q_target = q_target.detach()
        self.mask(q_pred, lengths)
        self.mask(q_target, lengths)
        loss = self.mse(q_pred, q_target)

        self.critic_optim.zero_grad()
        loss.backward()
        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        return loss

    def update(self, agents):
        # collect transistion memories form all agents
        memories = [a.memory for a in agents]

        # train model networks
        if self.use_agent_models:
            model_losses = []
            for _ in range(self.modeling_train_steps):
                batch = self.memory.sample_transitions_from(memories,
                                                            self.modeling_batch_size,
                                                            max_past=self.max_past)
                if self.obfuscate_others:
                    self.add_noise_(batch)
                model_losses.append(self.train_models(batch, agents).data)
            model_loss = np.mean(model_losses)
            model_kls = self.compare_models(agents, batch)
        else:
            model_loss = None
            model_kls = None

        # sample minibatch
        batch, lengths = self.memory.sample_episodes_from(memories, self.batch_size)
        if self.obfuscate_others:
            self.add_noise_(batch, lengths)
        # train actor and critic network
        actor_loss = self.train_actor(batch, lengths)
        critic_loss = self.train_critic(batch, agents, lengths)

        # update target network params
        self.update_params(self.actor_target, self.actor)
        self.update_params(self.critic_target, self.critic)

        return actor_loss, critic_loss, model_loss, model_kls
