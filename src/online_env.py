import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from src.blackjack.src.game import Game
import blackjack
from network import DQN
import math
import random
from collections import namedtuple, deque
from itertools import count
from online_manager import OnlineManager

class OnlineBlackjackEnv(gym.Env):
    def __init__(self):
        super(OnlineBlackjackEnv, self).__init__()

        self.game = OnlineManager()

        # Define the action and observation space
        # Assuming there are 'n' discrete actions like hit, stand, etc.
        n = 2  # 4 in reality but try with just hit and stand
        self.action_space = spaces.Discrete(n)  
        
        # Define the observation space according to your game state
        # Assuming a simple state representation as an example
        state_size = 15  # Replace with the actual size of the state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        self.game.new_game()


    def step(self, action):
        
        
        # basically deal first 
        # make necessary checks
        # ask for action
        # proceed and perhaps repeat the process
        print('Action:', action)
        new_state, reward, done, truncated, info = self.game.play_game(action)
        # print(new_state)
        # balances.append(info['balance'])
        array = np.array(new_state, dtype=object)
        flat_array = np.hstack(array)
        print("State:", flat_array)


        return flat_array, reward, done, truncated, info  # Additional info can be returned in the dictionary
    

    def reset(self):
        print("Starting new game:")
        # self.game.new_game()
        self.game.deal()
        new_state = self.game.get_state()
        # Explicitly create a numpy array from the integers, then concatenate the last list
        
        # Convert the list and nested list to NumPy arrays
        array = np.array(new_state, dtype=object)

        # Flatten the array
        flat_array = np.hstack(array)
        print("State:", flat_array)

        return flat_array, {}

    def render(self, mode='human'):
        # Implement this if you want to display the game state in a human-readable format
        pass

    def close(self):
        # Implement any cleanup if necessary
        pass    
    
