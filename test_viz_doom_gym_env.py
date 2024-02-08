import pytest
import numpy as np
from gymnasium.spaces import Discrete, Box
from backend.basic_agent.vizDoomGymEnv import VizDoomGym


class TestVizDoomGym:
    @pytest.fixture
    def env(self):
        return VizDoomGym(render=False)  # Set render=False for headless testing

    def test_observation_space(self, env):
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (100, 160, 1)
        assert env.observation_space.dtype == np.uint8

    def test_action_space(self, env):
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 3

    def test_reset(self, env):
        observation, info = env.reset()
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (100, 160, 1)
        assert info == {}

    def test_step(self, env):
        observation, reward, done, _, info = env.step(1)  # Assuming action 1 is valid
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (100, 160, 1)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "info" in info
