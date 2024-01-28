from vizdoom import *
import random
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import pygame

pygame.init()

game = DoomGame()
game.load_config('doomGame/VizDoom/scenarios/basic.cfg')
game.init()

actions = np.identity(3, dtype=np.uint8)


def takeinput():
    if pygame.key.get_pressed() == pygame.K_LEFT:
        return actions[0]
    if pygame.key.get_pressed() == pygame.K_RIGHT:
        return actions[1]
    if pygame.key.get_pressed() == pygame.K_SPACE:
        return actions[2]
    return [0, 0, 0]


episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        info = state.game_variables
        reward = game.make_action(takeinput())
        time.sleep(0.02)
    print('result', game.get_total_reward())
    time.sleep(2)
print(actions[0])
print(actions[1])
print(actions[2])
