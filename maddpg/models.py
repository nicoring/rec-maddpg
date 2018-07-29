import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Clonable:
    def clone(self, requires_grad=False):
        clone = copy.deepcopy(self)
        for param in clone.parameters():
            param.requires_grad = requires_grad
        return clone


class Actor(nn.Module, Clonable):
    @classmethod
    def from_actor(cls, actor):
        return cls(actor.n_inputs, actor.action_split, actor.n_hidden)

    def __init__(self, n_inputs, action_split, n_hidden):
        super().__init__()
        self.action_split = tuple(action_split)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = int(sum(action_split))
        self.init_layers()

    def init_layers(self):
        self.lin_1 = nn.Linear(self.n_inputs, self.n_hidden)
        self.lin_2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.lin_3 = nn.Linear(self.n_hidden, self.n_outputs)

    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        logits = self.lin_3(x)
        return logits

    def prob_dists(self, obs, temperature=1.0):
        logits = self.forward(obs)
        split_logits = torch.split(logits, self.action_split, dim=-1)
        temperature = torch.tensor(temperature).to(DEVICE)
        return [RelaxedOneHotCategorical(temperature, logits=l) for l in split_logits]

    def select_action(self, obs, explore=False, temp=1.0):
        distributions = self.prob_dists(obs, temp)
        if explore:
            actions = [d.rsample() for d in distributions]
        else:
            actions = [d.probs for d in distributions]
        return torch.cat(actions, dim=-1)

class LSTMActor(Actor):
    def init_layers(self):
        self.lin_input = nn.Linear(self.n_inputs, self.n_hidden)
        self.lstm = nn.LSTM(self.n_hidden, self.n_hidden)
        self.hidden2logits = nn.Linear(self.n_hidden, self.n_outputs)
        self.h_0 = nn.Parameter(torch.randn(self.n_hidden))
        self.c_0 = nn.Parameter(torch.randn(self.n_hidden))

    def init_state(self, batch_size):
        h_0 = self.h_0.repeat(1, batch_size, 1)
        c_0 = self.c_0.repeat(1, batch_size, 1)
        return (h_0, c_0)

    def forward(self, x):
        x = F.relu(self.lin_input(x))
        state = self.init_state(x.shape[1])
        x, _ = self.lstm(x, state)
        logits = self.hidden2logits(x)
        return logits


class Critic(nn.Module, Clonable):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, 1)

    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=-1)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        return self.lin_3(x)


class LSTMCritic(nn.Module, Clonable):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin = nn.Linear(n_inputs, n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden)
        self.hidden2value = nn.Linear(n_hidden, 1)
        self.h_0 = nn.Parameter(torch.randn(n_hidden))
        self.c_0 = nn.Parameter(torch.randn(n_hidden))

    def init_state(self, batch_size):
        h_0 = self.h_0.repeat(1, batch_size, 1)
        c_0 = self.c_0.repeat(1, batch_size, 1)
        return (h_0, c_0)

    def lstm_detached(self, x, state):
        x_ts = []
        for t in x:
            t = t.unsqueeze(0)
            x_t, (h, c) = self.lstm(t, state)
            x_ts.append(x_t)
            state = (h.detach(), c.detach())
        x = torch.cat(x_ts, dim=0)
        return x

    def forward(self, observations, actions, state=None, detached_states=False):
        x = torch.cat([*observations, *actions], dim=-1)
        x = F.relu(self.lin(x))
        if state is None:
            state = self.init_state(x.shape[1])
        if detached_states:
            x = self.lstm_detached(x, state)
        else:
            x, _ = self.lstm(x, state)
        values = self.hidden2value(x)
        return values
