import random
import numpy as np
from utils import MemTooSmallError

class EpsilonGreedyPolicy(object):
    def __init__(self, start_eps, eps_decay, n_actions, action_values):
        self.epsilon = start_eps
        self.eps_decay = eps_decay
        self.action_values = action_values
        self.possible_actions = range(n_actions)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions)
        else:
            state_action_values = self.action_values(state)
            return np.argmax(state_action_values)

    def update_epsilon(self):
        self.epsilon *= self.eps_decay
