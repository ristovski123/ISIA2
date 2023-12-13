import gymnasium
from stable_baselines3 import A2C
import numpy as np
import os

class TupleToMultiDiscreteWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(TupleToMultiDiscreteWrapper, self).__init__(env)
        parametro_1 = self.observation_space.spaces[0].n
        parametro_2 = self.observation_space.spaces[1].n
        parametro_3 = self.observation_space.spaces[2].n

        self.observation_space = gymnasium.spaces.MultiDiscrete([parametro_1,parametro_2,parametro_3])

    def observation(self, observation):
        parametro_1 = observation[0]
        parametro_2 = observation[1]
        parametro_3 = observation[2]
        observation_space = np.array([parametro_1,parametro_2,parametro_3])
        return observation_space
        # Convert Tuple observation to MultiDiscrete observation
        


models_dir = "models/OrigingalBlackjack/A2C"
logdir = "logs/OriginalEnv"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Example usage
env = gymnasium.make("Blackjack-v1",render_mode = "rgb_array")  # Replace with the actual name of your environment
env = TupleToMultiDiscreteWrapper(env)

observation, info = env.reset()
model = A2C('MlpPolicy', env, verbose=1,tensorboard_log=logdir)
# model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=10000)



TIMESTEPS = 10000
iters = 0
for i in range(1500):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


episodes = 10

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        # pass observation to model to get predicted action
        action, _states = model.predict(obs)
        print("obs: ", obs)
        print("action made by model: ", action)
        # pass action to env and get info back
        obs, rewards, done,trunc,  info = env.step(action)

        # show the environment on the screen
        env.render()
        print(ep, rewards, done)
        print("---------------")

# Now, env has a MultiDiscrete observation space
