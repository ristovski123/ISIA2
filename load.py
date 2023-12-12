import gymnasium as gym
from stable_baselines3 import PPO
from cona import TupleToMultiDiscreteWrapper


print ("cona pesada")
models_dir = "models/Blackjack/PPO"

env = gym.make('Blackjack-v1', render_mode="human")  # continuous: LunarLanderContinuous-v2
env = TupleToMultiDiscreteWrapper(env)
env.reset()

model_path = f"{models_dir}/290000.zip"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, trunc, done, info = env.step(action)
        env.render()