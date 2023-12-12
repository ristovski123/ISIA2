import gymnasium as gym
from gymnasium.wrappers import 
env = gym.make("Blackjack-v1", render_mode = "human")
observation, info = env.reset()
print(env.observation_space)
print(env.action_space)



for _ in range(1):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("Observation: ", observation)
    print("Reward: ", reward)
    print("action: ", action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
env.close()