import gymnasium as gym
from stable_baselines3 import PPO
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
models_dir = "models/Blackjack/PPO"

env = gym.make('Blackjack-v1', render_mode="rgb_array")  # continuous: LunarLanderContinuous-v2
env = TupleToMultiDiscreteWrapper(env)
env.reset()

model_path = f"{models_dir}/1210000.zip"
model = PPO.load(model_path, env=env)

episodes = 100000
jogadas = []

for ep in range(episodes):
    jogada = []
    obs, info = env.reset()
    done = False

    while not done:
        # Pass observation to model to get predicted action
        action, _states = model.predict(obs)

        # Store the current step's information
        step_info = {'obs': obs, 'action': action}
        jogada.append(step_info)

        # Pass action to the environment and get info back
        obs, rewards, done, trunc, info = env.step(action)

    # Store the final step's information when the episode is done
    step_info = {'obs': obs, 'rewards': rewards}
    jogada.append(step_info)

    # Append the entire episode's trajectory to the list
    jogadas.append(jogada)

# Now, jogadas contains the trajectory for each episode


    env.render()

#print("caBOU")
#print(jogadas)

import pandas as pd

# Assuming jogadas is a list of lists (each sublist represents an episode's trajectory)
# where each item in the sublist is a dictionary containing 'obs', 'action', 'rewards', etc.

# Flatten the list of dictionaries into a list of flat dictionaries
flat_jogadas = [step_info for episode in jogadas for step_info in episode]

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(flat_jogadas)

# Print or further analyze the DataFrame
print(df)

#move to csv

df.to_csv('jogadas.csv', index=False)

