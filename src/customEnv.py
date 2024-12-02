import random

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        # 0 = lower temp, 1 = keep temp, 2 = increase temp
        self.state += action - 1
        self.shower_length -= 1

        # Calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        # Check if shower is done
        done = self.shower_length <= 0

        # Apply temperature noise
        self.state += random.randint(-1, 1)
        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return self.state
