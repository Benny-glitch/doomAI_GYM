import unittest
from gymnasium.spaces import Box, Discrete
import numpy as np
from backend.agent.vizDoomGymEnv import VizDoomGym


class TestVizDoomGym(unittest.TestCase):

    def setUp(self):
        # Inizializzazione della classe VizDoomGym per ogni test
        self.env = VizDoomGym(render=False)  # Imposta render=False per eseguire i test senza renderizzazione della
        # finestra

    def tearDown(self):
        # Cleanup dopo ogni test, se necessario
        self.env.close()

    def test_observation_space(self):
        # Verifica che lo spazio delle osservazioni sia correttamente configurato come Box di dimensioni (100, 160, 1)
        self.assertTrue(np.all(self.env.observation_space.low == 0))
        self.assertTrue(np.all(self.env.observation_space.high == 255))
        self.assertEqual(self.env.observation_space.shape, (100, 160, 1))

    def test_action_space(self):
        # Verifica che lo spazio delle azioni sia correttamente configurato come Discrete con 3 azioni possibili
        self.assertTrue(isinstance(self.env.action_space, Discrete))
        self.assertEqual(self.env.action_space.n, 3)

    def test_reset(self):
        # Verifica che il metodo reset restituisca correttamente le osservazioni e le informazioni
        observations, info = self.env.reset()
        self.assertTrue(isinstance(observations, np.ndarray))
        self.assertEqual(observations.shape, (100, 160, 1))
        self.assertTrue(isinstance(info, dict))

    def test_step(self):
        # Verifica che il metodo step restituisca correttamente lo stato, la ricompensa, se l'episodio è terminato e
        # le informazioni
        _, _, done, _, info = self.env.step(0)  # Esegui un'azione fittizia (es. 0) per il test
        self.assertTrue(isinstance(done, bool))
        self.assertTrue(isinstance(info, dict))

        # Verifica che lo stato restituito abbia le dimensioni corrette e sia nell'intervallo corretto
        state, _, _, _, _ = self.env.step(0)  # Esegui un'altra azione fittizia per ottenere uno stato
        self.assertEqual(state.shape, (100, 160, 1))
        self.assertTrue(np.all(state >= 0) and np.all(state <= 255))


if __name__ == '__main__':
    unittest.main()
