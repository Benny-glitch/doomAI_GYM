from vizdoom import *
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2
import utils


# GrayScale the game frame and resize
def grayscale(observation):
    gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
    state = np.reshape(resize, (100, 160, 1))
    return state


class VizDoomGym(Env):
    # Method called for the init of the Env
    def __init__(self, render=False, scenarios=utils.BASIC_CONFIG_PATH):
        # Inherit from Env
        super().__init__()
        # Set up the game
        self.game = DoomGame()
        self.game.load_config(scenarios)

        # Bool Method game render or not
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()

        # Creation of the action and the observation space
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {"info": ammo}
        else:
            state = np.zeros(self.observation_space.shape)
            info = {"info": 0}

        done = self.game.is_episode_finished()

        return state, reward, done, False, info

    def close(self):
        self.game.close()

    # Method called from reset the Env
    def reset(self, **kwargs):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        observations = grayscale(state)
        info = {}  # Usa un dizionario vuoto se non hai informazioni di reset da restituire
        return observations, info
