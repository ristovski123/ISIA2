import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO


class CustomWrapper(Wrapper):
    def __init__(self,env):
        super(CustomWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info


    
    def observation(self, observation):
        retorno = tuple[int,int,bool]
        retorno = (observation[0], observation[1], observation[2])
        return retorno
    

env = gym.make("Blackjack-v1", render_mode = "rgb_array")
env = CustomWrapper(env)
observation, info = env.reset()
model = PPO('MlpPolicy', env, verbose=1)
# model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100)


