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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()

        self.game = blackjack.Manager()
        print("Your balance is:", self.game.players[0].balance)
        self.balances = []
        self.balances.append(self.game.players[0].balance)

        # Define the action and observation space
        # Assuming there are 'n' discrete actions like hit, stand, etc.
        n = 2  # 4 in reality but try with just hit and stand
        self.action_space = spaces.Discrete(n)  
        
        # Define the observation space according to your game state
        # Assuming a simple state representation as an example
        state_size = 15  # Replace with the actual size of the state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

    def step(self, action):
        
        
        # basically deal first 
        # make necessary checks
        # ask for action
        # proceed and perhaps repeat the process
        # return the new state, reward, and whether the game is done
        print(action)
        new_state, reward, done, truncated, info = self.game.play_game(action)
        print(new_state)
        # print(new_state)
        # balances.append(info['balance'])
        integers = np.array(new_state[:-1], dtype=np.float32)  # Convert the integer elements
        last_list = np.array(new_state[-1], dtype=np.float32)  # Convert the last element which is a list
        flattened_state = np.concatenate((integers, last_list))  # Concatenate both arrays


        return flattened_state, reward, done, truncated, info  # Additional info can be returned in the dictionary
    

    def reset(self):
        print("NEW GAME")
        new_state, info = self.game.new_game()
        print("State:", new_state)
        # print("Your balance is:", self.game.players[0].balance)
        self.balances.append(self.game.players[0].balance)
        # Explicitly create a numpy array from the integers, then concatenate the last list
        integers = np.array(new_state[:-1], dtype=np.float32)  # Convert the integer elements
        last_list = np.array(new_state[-1], dtype=np.float32)  # Convert the last element which is a list
        flattened_state = np.concatenate((integers, last_list))  # Concatenate both arrays

        return flattened_state, info

    def render(self, mode='human'):
        # Implement this if you want to display the game state in a human-readable format
        pass

    def close(self):
        # Implement any cleanup if necessary
        pass    
    
