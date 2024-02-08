import os
import pytest
from stable_baselines3 import PPO
from backend.basic_agent.vizDoomGymEnv import VizDoomGym
from backend.basic_agent.gameTrainer import TrainAndLoggingCallback

CHECKPOINT_DIR = '../train/train_test'
LOG_DIR = '../logs/log_test'
n_steps = 2
n_envs = 1
batch_size = n_steps * n_envs


class TestTraining:
    @pytest.fixture
    def callback(self):
        return TrainAndLoggingCallback(check_freq=10, save_path=CHECKPOINT_DIR)

    @pytest.fixture
    def env(self):
        return VizDoomGym(render=False)

    def test_init(self, callback):
        # Verifica che i parametri vengano inizializzati correttamente
        assert callback.check_freq == 10
        assert callback.save_path == "../train/train_test"

    def test_init_callback(self, callback, tmpdir):
        # Verifica che la directory di salvataggio venga creata correttamente
        callback.save_path = str(tmpdir.join("../train/train_test"))
        callback._init_callback()
        assert os.path.exists(callback.save_path)

    def test_start_training(self, callback, env):
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=n_steps, batch_size=batch_size)
        model.learn(total_timesteps=8, callback=callback)

        assert os.path.exists(CHECKPOINT_DIR)  # Check if the checkpoint directory was created
        assert os.path.exists(os.path.join(CHECKPOINT_DIR, 'best_model_10.zip'))  # Check if a model file was saved
