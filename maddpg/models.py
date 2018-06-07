import math
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
    def __init__(self, n_inputs, n_outputs, n_hidden):
        super().__init__()
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = self.lin_3(x)
        return x

    def act_det(self, x):
        return F.softmax(self.forward(x), dim=-1)

    def act_rand(self, x):
        logits = self.forward(x)
        u = torch.rand_like(logits)
        return F.softmax(logits - torch.log(-torch.log(u)), dim=-1).detach()


class Critic(nn.Module, Clonable):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, 1)
    
    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=1)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        return self.lin_3(x)
