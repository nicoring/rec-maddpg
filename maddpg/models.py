import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class Clonable:
    def clone(self):
        clone = copy.deepcopy(self)
        for param in clone.parameters():
            param.requires_grad = False
        return clone


class Actor(nn.Module, Clonable):
    def __init__(self, n_inputs, action_split, n_hidden):
        super().__init__()
        self.action_split = tuple(action_split)
        n_outputs = int(sum(action_split))
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = self.lin_3(x)
        splits_logits = torch.split(x, self.action_split, dim=-1)
        splits_actions = [F.softmax(s, dim=-1) for s in splits_logits]
        return torch.cat(splits_actions, dim=-1)


class Critic(nn.Module, Clonable):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, 1)

    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=-1)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        return self.lin_3(x)
