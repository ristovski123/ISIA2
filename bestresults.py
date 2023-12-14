import gymnasium as gym
from stable_baselines3 import PPO, A2C
import numpy as np

class TupleToMultiDiscreteWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TupleToMultiDiscreteWrapper, self).__init__(env)
        parametro_1 = self.observation_space.spaces[0].n
        parametro_2 = self.observation_space.spaces[1].n
        parametro_3 = self.observation_space.spaces[2].n

        self.observation_space = gym.spaces.MultiDiscrete([parametro_1,parametro_2,parametro_3])

    def observation(self, observation):
        parametro_1 = observation[0]
        parametro_2 = observation[1]
        parametro_3 = observation[2]
        observation_space = np.array([parametro_1,parametro_2,parametro_3])
        return observation_space
        # Convert Tuple observation to MultiDiscrete observation
        

print ("cona pesada")
models_dir = "models/OriginalBlackjack/PPO"

env = gym.make('Blackjack-v1', render_mode="rgb_array")  # continuous: LunarLanderContinuous-v2
env = TupleToMultiDiscreteWrapper(env)
env.reset()

model_path = f"{models_dir}/7860000.zip"
model = PPO.load(model_path, env=env)

episodes = 10000
rewards1 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards1.append(reward)

model_path = f"{models_dir}/7870000.zip"
model = PPO.load(model_path, env=env)

episodes = 10000
rewards2 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards2.append(reward)


model_path = f"{models_dir}/2090000.zip"
model = PPO.load(model_path, env=env)

episodes = 10000
rewards3 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards3.append(reward)

model_path = f"{models_dir}/2100000.zip"
model = PPO.load(model_path, env=env)

episodes = 10000
rewards4 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards4.append(reward)

models_dir = "models/OrigingalBlackjack/A2C"

model_path = f"{models_dir}/6710000.zip"
model = A2C.load(model_path, env=env)

episodes = 10000
rewards5 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards5.append(reward)



model_path = f"{models_dir}/6700000.zip"
model = A2C.load(model_path, env=env)

episodes = 10000
rewards6 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards6.append(reward)

model_path = f"{models_dir}/8860000.zip"
model = A2C.load(model_path, env=env)

episodes = 10000
rewards7 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards7.append(reward)

model_path = f"{models_dir}/8870000.zip"
model = A2C.load(model_path, env=env)

episodes = 10000
rewards8 = []
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
    rewards8.append(reward)

# compare all the reward mean values to see what the best model is
print(len(rewards1))


def load_modelppo(model_path,flag):
    if flag == 1:
        model = PPO.load(model_path, env=env)
        print("PPO",model_path)
    else:
        model = A2C.load(model_path, env=env)
        print("A2C",model_path)

    episodes = 10000
    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)
        rewards.append(reward)
    return rewards

#now i want to run this 20 times so in the end I can make a mean of each model ( ppo1,ppo2,...) and pick the best one

dataset = []


for i in range (20):
    models_dir = "models/OriginalBlackjack/PPO"
    ppo1 = load_modelppo(f"{models_dir}/7860000.zip",1)
    ppo2 = load_modelppo(f"{models_dir}/7870000.zip",1)
    ppo3 = load_modelppo(f"{models_dir}/2090000.zip",1)
    ppo4 = load_modelppo(f"{models_dir}/2100000.zip",1)
    models_dir = "models/OrigingalBlackjack/A2C"
    a2c1 = load_modelppo(f"{models_dir}/6710000.zip",2)
    a2c2 = load_modelppo(f"{models_dir}/6700000.zip",2)
    a2c3 = load_modelppo(f"{models_dir}/8860000.zip",2)
    a2c4 = load_modelppo(f"{models_dir}/8870000.zip",2)
    iteration_models = [ppo1, ppo2, ppo3, ppo4, a2c1, a2c2, a2c3, a2c4]
    dataset.append(iteration_models)

#save dataset to csv

import csv

with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["PPO1", "PPO2", "PPO3", "PPO4", "A2C1", "A2C2", "A2C3", "A2C4"])
    for i in range(len(dataset)):
        writer.writerow(dataset[i])

