import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import backend.agent.vizDoomGymEnv as vizDoomGymEnv

CHECKPOINT_DIR = '../../train/train_basic'
LOG_DIR = '../../logs/log_basic'


def start_training():
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    env = vizDoomGymEnv.VizDoomGym(render=True)
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
    model.learn(total_timesteps=100000, callback=callback)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

