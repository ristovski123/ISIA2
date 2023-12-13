import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TransformObservation

env = gym.make("Blackjack-v1", render_mode = "human")
env = TransformObservation(env, lambda obs: Boxobs + 0.1)
env.reset()

model = PPO('MlpPolicy', env, verbose=1)
# model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100)

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
        obs, rewards, trunc, done, info = env.step(action)

        # show the environment on the screen
        env.render()
        print(ep, rewards, done)
        print("---------------")