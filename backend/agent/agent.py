import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from vizDoomGymEnv import VizDoomGym

# Reload model from disc
model = PPO.load('../../train/train_basic/best_model_100000.zip')
# Create rendered environment
original_env = DummyVecEnv([lambda: VizDoomGym(render=True)])

# Reset dell'ambiente
obs = original_env.reset()

for episode in range(100):
    obs = original_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = original_env.step(action)
        time.sleep(0.02)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(0.5)
