import gymnasium
from stable_baselines3 import PPO
from blackjenv import BlackjackWithDoubleEnv
import numpy as np
import os


class TupleToMultiDiscreteWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(TupleToMultiDiscreteWrapper, self).__init__(env)
        parametro_1 = self.observation_space.spaces[0].n
        parametro_2 = self.observation_space.spaces[1].n
        parametro_3 = self.observation_space.spaces[2].n

        self.observation_space = gymnasium.spaces.MultiDiscrete([parametro_1, parametro_2, parametro_3])

    def observation(self, observation):
        parametro_1 = observation[0]
        parametro_2 = observation[1]
        parametro_3 = observation[2]
        observation_space = np.array([parametro_1, parametro_2, parametro_3])
        return observation_space
        # Convert Tuple observation to MultiDiscrete observation


models_dir = "models/OriginalBlackjack/PPO_model2"

env = BlackjackWithDoubleEnv()
env = TupleToMultiDiscreteWrapper(env)
env.reset()
model_path = f"{models_dir}/8160000.zip"
model = PPO.load(model_path, env=env)

episodes = 100000
rewards = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards.append(reward)

rewardsmean = np.mean(rewards)

winrate = 0
for i in rewards:
    if i == 1:
        winrate += 1

winrate = winrate/episodes*100
print("Winrate: ",winrate)