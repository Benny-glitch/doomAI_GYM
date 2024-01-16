from vizdoom import *
import random
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

class VizDoomGym():
    # Method called for the init of the Env
    def __init__(self, render=False):
        self.game = DoomGame()
        self.game.load_config('doomGame/VizDoom/scenarios/basic.cfg')
        self.game.init()

        #Bool Method game render or not
        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Creation of the action and the obsservation space
        self.observation_space = Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action])

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        done = self.game.is_episode_finished()

        return state, reward, info, done
    def close(self):
        self.game.close()

    def render(self):
        pass

    def grayScale(self):
        pass

    def resetEnv(self):
        pass

env = VizDoomGym(render=True)
env.step(2)