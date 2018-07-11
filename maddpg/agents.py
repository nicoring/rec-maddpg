import numpy as np
import torch

from memory import ReplayBuffer
import distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    def act(self, obs):
        raise NotImplementedError

    def experience(self, obs, action, reward, new_obs, done):
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
        action = np.zeros(4)
        for a in acc:
            if a >= 0:
                action[0] = a
            else:
                action[1] = -a
        if abs(np.sum(action)) > 0:
            action = action / np.sum(action)
        return action

    def act(self, obs):
        # vel = obs[:2]
        l1 = obs[2:4]
        l2 = obs[4:6]
        l3 = obs[6:8]
        # a1 = obs[8:10]
        # a2 = obs[10:12]
        # target = self.get_target([l1, l2, l3], [a1, a2])
        landmarks = [l1, l2, l3]
        dists = [self.length(l) for l in landmarks]
        target = landmarks[np.argmin(dists)]
        return self.acc2action(target)


class RandomAgent(Agent):
    def __init__(self, index, name, env):
        self.env = env
        self.index = index
        self.name = name
        self.num_actions = self.env.action_space[self.index].n

    def act(self, obs):
        logits = np.random.sample(self.num_actions)
        return logits / np.sum(logits)


class MaddpgAgent(Agent):
    def __init__(self, index, name, actor, critic, params):
        self.index = index
        self.name = name

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = actor.clone().to(device)
        self.critic_target = critic.clone().to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=params.lr_critic)
        self.memory = ReplayBuffer(params.memory_size)
        self.mse = torch.nn.MSELoss()

        # params
        self.batch_size = params.batch_size
        self.tau = params.tau
        self.gamma = params.gamma
        self.clip_grads = True
        self.sigma = params.sigma

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

    def init_agent_models(self, agents):
        for agent in agents:
            if agent is self:
                continue
            agent_model = agent.actor.clone(requires_grad=True).to(device)
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
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(device)
        actions = self.actor.select_action(obs, explore=explore).detach()
        return actions.to('cpu')

    def experience(self, obs, action, reward, new_obs, done):
        self.memory.add(obs, action, reward, new_obs, float(done))

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
                loss = (-dist.log_prob(actions)).mean() # + self.entropy_weight * dist.entropy()
                losses[i] = loss
            loss = torch.mean(losses)
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

    def update(self, agents):
        # sample minibatch
        memories = [a.memory for a in agents]

        # train networks
        if self.use_agent_models:
            model_losses = []
            for _ in range(self.modeling_train_steps):
                batch = ReplayBuffer.sample_from_memories(memories,
                                                          self.modeling_batch_size,
                                                          max_past=self.max_past)
                model_losses.append(self.train_models(batch, agents).data)
            model_loss = np.mean(model_losses)
            model_kls = self.compare_models(agents, batch)
        else:
            model_loss = None
            model_kls = None

        batch = ReplayBuffer.sample_from_memories(memories, self.batch_size)
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
        }, self.memory.memory, model_pair

    def load_state(self, state):
        for key, value in state['state_dicts'].items():
            getattr(self, key).load_state_dict(value)
        if 'memory' in state:
            self.memory.memory = state['memory']
        if 'models' in state:
            models, optims = state['models']
            for i, m in models.items():
                self.agent_models[i].load_state_dict(m)
            for i, o in optims.items():
                self.model_optims[i].load_state_dict(o)
