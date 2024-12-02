import gymnasium
import pandas
import numpy
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR


def demo():
    episodes = 5
    env = gymnasium.make("FrozenLake-v1", render_mode="human")
    lifeMemory = []

    for i in range(episodes):
        observation = env.reset()
        done = False
        totalReward = 0
        episodeMemory = []
        while not done:
            newAction = env.action_space.sample()
            newObservation, reward, done, truncated, info = env.step(newAction)
            totalReward += reward

            episodeMemory.append({
                "observation": observation,
                "action": newAction,
                "reward": reward,
                "episode": i,
            })
            observation = newObservation

        steps = len(episodeMemory)
        for i, memory in enumerate(episodeMemory):
            memory["totalReward"] = totalReward
            memory["decayReward"] = i * totalReward / steps

        lifeMemory.extend(episodeMemory)

    memoryDF = pandas.DataFrame(lifeMemory)
    print(memoryDF.describe())
    print(memoryDF.shape)
    print(memoryDF.groupby("episode").reward.sum().mean())

    model = ExtraTreesRegressor(n_estimators=50)
    # model = SVR()
    y = 0.5*memoryDF.reward + 0.1*memoryDF.decayReward + memoryDF.totalReward
    x = memoryDF[["observation", "action"]]
    model.fit(x, y)
