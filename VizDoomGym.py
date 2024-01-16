from vizdoom import *
import random
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2


class VizDoomGym(Env):
    # Method called for the init of the Env
    def __init__(self, render=False):
        # Inherit from Env
        super().__init__()
        #Setup the game
        self.game = DoomGame()
        self.game.load_config('doomGame/VizDoom/scenarios/basic.cfg')
        self.game.init()

        # Bool Method game render or not
        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Creation of the action and the obsservation space
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action])

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayScale(state)
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        done = self.game.is_episode_finished()

        return state, reward, info, done

    def close(self):
        self.game.close()

    # Method called from reset the Env
    def resetEnv(self):
        self.game.new_episode()
        state = self.game.new_episode()
        return self.grayScale(state)

    # GrayScale the game frame and resize
    def grayScale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, 1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, -1, (100, 160, 1))

        return state
