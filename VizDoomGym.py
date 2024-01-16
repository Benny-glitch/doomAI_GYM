from vizdoom import *
import random
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
class VizDoomGym():
    #Method called for the init of the Env
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config('doomGame/VizDoom/scenarios/basic.cfg')
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(240, 320 , 3), dtype=np.uint8)
    def step(self, action):
        pass
    def close():
        pass
    def render():
        pass
    def grayScale():
        pass
    def resetEnv():
        pass
