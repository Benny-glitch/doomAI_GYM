import os
import shutil
import unittest
from stable_baselines3 import PPO
from backend.agent.gameTrainer import TrainAndLoggingCallback
from backend.agent.vizDoomGymEnv import VizDoomGym


class TestTrainAndLoggingCallback(unittest.TestCase):

    def setUp(self):
        # Creazione di una directory temporanea per i controlli e i log
        self.checkpoint_dir = './tmp_train'
        self.log_dir = './tmp_logs'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def tearDown(self):
        # Pulizia della directory temporanea
        shutil.rmtree(self.checkpoint_dir)
        shutil.rmtree(self.log_dir)

    def test_callback(self):
        # Verifica che il callback salvi il modello ogni check_freq iterazioni
        check_freq = 10
        callback = TrainAndLoggingCallback(check_freq=check_freq, save_path=self.checkpoint_dir)

        env = VizDoomGym(render=False)
        model = PPO('CnnPolicy', env, tensorboard_log=self.log_dir, verbose=1,
                    learning_rate=0.0001, n_steps=2)

        model.learn(total_timesteps=10, callback=callback)

        # Verifica che il modello venga salvato almeno una volta
        self.assertTrue(os.path.exists(self.checkpoint_dir))

        # Verifica che il modello venga salvato solo ogni check_freq iterazioni
        self.assertEqual(len(os.listdir(self.checkpoint_dir)), 1)


if __name__ == '__main__':
    unittest.main()
