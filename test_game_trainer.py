import os
import numpy as np
import pytest
from stable_baselines3 import PPO
from backend.agent.vizDoomGymEnv import VizDoomGym
from backend.agent.gameTrainer import TrainAndLoggingCallback

CHECKPOINT_DIR = '../train/train_test'
LOG_DIR = '../logs/log_test'


class TestTraining:
    @pytest.fixture
    def callback(self):
        return TrainAndLoggingCallback(check_freq=10, save_path=CHECKPOINT_DIR)

    @pytest.fixture
    def env(self):
        return VizDoomGym(render=False)

    def test_start_training(self, callback, env):
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2)
        model.learn(total_timesteps=10, callback=callback)

        assert os.path.exists(CHECKPOINT_DIR)  # Check if the checkpoint directory was created
        assert os.path.exists(os.path.join(CHECKPOINT_DIR, 'best_model_10.zip'))  # Check if a model file was saved
