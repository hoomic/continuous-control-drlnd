import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[1024, 512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list(int)): Number of nodes in hidden_layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.layers = nn.ModuleList([nn.Linear(fc_units[i], fc_units[i+1]) for i in range(len(fc_units) - 1)])
        self.output = nn.Linear(fc_units[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for l in self.layers:
            l.weight.data.uniform_(*hidden_init(l))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        for l in self.layers:
            x = F.relu(l(x))
        return F.tanh(self.output(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units= [1024, 512], action_cat_layer=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list(int)): Number of nodes in hidden_layers
            action_cat_layer (int): Index of hidden layers to concatenate actions
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_cat_layer = action_cat_layer
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.layers = nn.ModuleList([nn.Linear(fc_units[i] + (action_size if i == action_cat_layer else 0), fc_units[i+1]) for i in range(len(fc_units) - 1)])
        self.output = nn.Linear(fc_units[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for l in self.layers:
            l.weight.data.uniform_(*hidden_init(l))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        for i, l in enumerate(self.layers):
            if i == self.action_cat_layer:
                x = torch.cat((x, action), dim=1)
            x = F.relu(l(x))
        return self.output(x)
