import gymnasium as gym
from gymnasium import spaces
import numpy as np
from blackjack import Manager
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

# class BlackjackEnv(gym.Env):
#     def __init__(self):
#         super(BlackjackEnv, self).__init__()

#         self.game = Manager()
#         self.balances = []
#         self.balances.append(self.game.player.balance)

#         # Define the action and observation space
#         # Assuming there are 'n' discrete actions like hit, stand, etc.
#         n = 2  # 4 in reality but try with just hit and stand
#         self.action_space = spaces.Discrete(n)  
        
#         # Define the observation space according to your game state
#         # Assuming a simple state representation as an example
#         state_size = 15  # Replace with the actual size of the state
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

#     def step(self, action):
        
        
#         # basically deal first 
#         # make necessary checks
#         # ask for action
#         # proceed and perhaps repeat the process
#         # return the new state, reward, and whether the game is done
#         print(action)
#         new_state, reward, done, truncated, info = self.game.play_game(action)
#         print(new_state)
#         integers = np.array(new_state[:-1], dtype=np.float32)  
#         last_list = np.array(new_state[-1], dtype=np.float32)  
#         flattened_state = np.concatenate((integers, last_list))

#         return flattened_state, reward, done, truncated, info  # Additional info can be returned in the dictionary
    

#     def reset(self):
#         print("NEW GAME")
#         new_state, info = self.game.new_game(100)
#         print("State:", new_state)
#         self.balances.append(self.game.player.balance)
#         integers = np.array(new_state[:-1], dtype=np.float32)  # Convert the integer elements
#         last_list = np.array(new_state[-1], dtype=np.float32)  # Convert the last element which is a list
#         flattened_state = np.concatenate((integers, last_list))  # Concatenate both arrays

#         return flattened_state, info

#     def render(self, mode='human'):
#         # Implement this if you want to display the game state in a human-readable format
#         pass

#     def close(self):
#         # Implement any cleanup if necessary
#         pass   
    
    
class BlackjackEnv(gym.Env):
    """Custom Blackjack environment based on OpenAI's Gymnasium."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, balance=100000):
        super(BlackjackEnv, self).__init__()
        
        self.manager = Manager(balance)
        self.balance = balance
        self.balances = [balance]
        self.action_space = spaces.Discrete(2)
        
        state_size = 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        
        self.state = None
        self.done = False

    def reset(self):
        """Resets the game state for a new round."""
        bet = 10  # Starting bet (could be made dynamic based on action)
        self.state, game_over = self.manager.new_game(bet)
        self.done = game_over

        self.balances.append(self.manager.player.balance)
        
        return self.state, {}

    # def _get_obs(self):
    #     """Helper function to get the current state (observation)."""
    #     player_hand_value = self.state[0]
    #     dealer_visible_card = self.manager.dealer.hand.cards[0]
    #     return np.array([player_hand_value, dealer_visible_card, self.manager.player.balance])
    
    def step(self, action):
        """Takes a step in the game based on the action provided."""
        # if self.done:
        #     raise ValueError("Game is over. Please reset the environment.")

        state, reward, done, truncated, info = self.manager.play_game(action)

        # print("state:", state)
        # print(len(state))
        self.state = state
        self.done = done
        
        # Get the observation
        # obs = self._get_obs()

        return state, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Renders the current state of the game to the console."""
        player_hand_value = self.state[0]
        player_cards = self.manager.player.hands[0].cards
        dealer_visible_card = self.manager.dealer.hand.cards[0]

        print(f"\nYour Hand: {format_hand(player_cards)} (Value: {player_hand_value})")
        print(f"Dealer's Visible Card: {dealer_visible_card}")
        print(f"Player Balance: {self.manager.player.balance}\n")
    
    def close(self):
        """Cleans up any resources when the environment is closed."""
        pass

def format_hand(cards):
    """Utility function to format the hand as a readable string."""
    return ', '.join(str(card) for card in cards)
 
    
