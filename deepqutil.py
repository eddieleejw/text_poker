'''
Holds misc classes and functions that the DQNAgent would use
'''


import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition (which is a named tuple)'''

        t = Transition(*args)

        state = t.state
        assert(state is None or (state.dim() == 2 and state.shape[0] == 1 and state.shape[1] == 113))


        action = t.action
        assert(action.dim() == 2)
        assert(action.shape[0] == 1 and action.shape[1] == 1)
        assert(type(action[0][0].item()) is int)

        next_state = t.next_state
        assert(next_state is None or (next_state.dim() == 2 and next_state.shape[0] == 1 and next_state.shape[1] == 113))
        
        reward = t.reward
        # reward should have shape [1] i.e. look like tensor([float])
        assert(reward.dim() == 1)
        assert(reward.shape[0] == 1)

        self.memory.append(t)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(n_observations, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

