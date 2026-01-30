# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:08:17 2026

@author: USER
"""

# =============================================================================
# Environment
#    ↑        ↓
#  state   reward
#    ↑        ↓
#   Agent —— action
# 
# =============================================================================

# =============================================================================
# for episode:
#     reset environment
#     for step:
#         choose action
#         step environment
#         update agent
# =============================================================================


class Env:
    def reset(self):
        """
        回傳初始 state
        """
        raise NotImplementedError

    def step(self, action):
        """
        輸入 action
        回傳:
            next_state
            reward
            done
        """
        raise NotImplementedError

class LineWorld(Env):
    def __init__(self, size=5):
        self.size = size
        self.goal = size - 1

    def reset(self):
        self.pos = 0
        return self.pos

    def step(self, action):
        # action: 0 = left, 1 = right
        if action == 1:
            self.pos = min(self.pos + 1, self.goal)
        else:
            self.pos = max(self.pos - 1, 0)

        reward = 1.0 if self.pos == self.goal else -0.01
        done = (self.pos == self.goal)

        return self.pos, reward, done

import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions,
                 lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state):
        # ε-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        target = r
        if not done:
            target += self.gamma * np.max(self.Q[s_next])

        self.Q[s, a] += self.lr * (target - self.Q[s, a])
