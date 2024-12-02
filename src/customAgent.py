import numpy as np
from keras import models, layers, optimizers

from src.customEnv import ShowerEnv


def buildModel(states, actions):
    model = models.Sequential()
    model.add(layers.Dense(24, activation="relu", input_shape=states))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))
    return model


def customAgent():
    env = ShowerEnv()
    states = env.observation_space.shape
    actions = env.action_space.n
    del model
    model = buildModel(states, actions)
    # model.summary()

    # episodes = 10
    # for episode in range(1, episodes + 1):
    #     state = env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         # env.render()
    #         action = env.action_space.sample()
    #         n_state, reward, done, info = env.step(action)
    #         score += reward
    #     print(f"Episode: {episode}, Score: {score}")
