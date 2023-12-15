import gymnasium
from stable_baselines3 import PPO
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
        


models_dir = "models/OriginalBlackjack/PPOTunned"
logdir = "logs/TunnedEnv"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Example usage
env = gymnasium.make("Blackjack-v1", render_mode = "rgb_array")  # Replace with the actual name of your environment
env = TupleToMultiDiscreteWrapper(env)

observation, info = env.reset()
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=10000)



TIMESTEPS = 10000
iters = 0
for i in range(1500):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# Now, env has a MultiDiscrete observation space
