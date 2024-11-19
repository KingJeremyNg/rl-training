import os

import ale_py
import gymnasium
from stable_baselines3 import A2C, DQN, PPO


def stableAgent():
    savePath = "./models/125x125_gray_PPO_Mlp.zip"
    timesteps = 10000000
    observationSpace = (125, 125)

    gymnasium.register_envs(ale_py)
    env = gymnasium.make("ALE/Centipede-v5")
    # achieve an nxn square for the observation space
    env = gymnasium.wrappers.ResizeObservation(env, observationSpace)
    # convert to grayscale keeping the "color channel" dimension for now
    env = gymnasium.wrappers.GrayscaleObservation(env, keep_dim=True)

    if os.path.exists(savePath):
        print(f"Loading model from {savePath}")
        model = PPO.load(savePath, env, verbose=1)
    else:
        print("Creating new model")
        model = PPO("MlpPolicy", env, verbose=1)

    # will train model over timesteps and report episode parameters
    model.learn(timesteps)
    print(f"Saving model to {savePath}")
    model.save(savePath)

    vec_env = model.get_env()
    obs = vec_env.reset()
    terminated = False
    rewards = 0
    while not terminated:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, info = model.env.step(action)
        rewards += reward
        print(rewards)

    env.close()
