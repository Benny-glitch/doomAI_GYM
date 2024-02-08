from stable_baselines3 import PPO
import vizDoomGymEnv as viz_doom_gym_env
import os

CHECKPOINT_DIR = '/usr/share/app/train/train_basic'
LOG_DIR = '/usr/share/app/logs/log_basic'

env = viz_doom_gym_env.VizDoomGym(render=False)
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
model.learn(total_timesteps=50000)
save_path = '/usr/share/app/model.zip'
if os.path.exists(os.path.join('/usr/share/app', 'model.zip')):
    old_file = os.path.join('/usr/share/app', 'model.zip')
    new_file = os.path.join('/usr/share/app', 'model_first_50000.zip')
model.save(save_path)

