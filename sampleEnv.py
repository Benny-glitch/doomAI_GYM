from vizdoom import *
import random
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2

game = DoomGame()
game.load_config('doomGame/VizDoom/scenarios/basic.cfg')
game.init()

actions = np.identity(3, dtype=np.uint8)

episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        info = state.game_variables
        reward = game.make_action(random.choice(actions))
        print('reward:', reward)
        print('info', info)
        time.sleep(0.02)
    print('result', game.get_total_reward())
    time.sleep(2)