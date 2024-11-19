import gymnasium
import ale_py

gymnasium.register_envs(ale_py)
env = gymnasium.make("ALE/Asteroids-v5", render_mode="human")

observation, info = env.reset(seed=42)

spin_fire_actions = [1, 3]  # 1 == FIRE, 3 == RIGHT

for step_index in range(1000):
    action = spin_fire_actions[step_index % 2]
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
