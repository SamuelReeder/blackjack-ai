import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from environment import BlackjackEnv
from model import DQN

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f'Using {torch.cuda.get_device_name(0)}\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 2:
    print("Please provide the model number to load")
    sys.exit(1)
num = sys.argv[1]

num_episodes = 1000 
cumulative_rewards = []
game_lengths = []

env = BlackjackEnv()
starting_balance, starting_bet = env.balance, env.bet
n_actions = env.action_space.n
n_observations = env.state_size

policy_net = DQN(n_observations, n_actions).to(device)

policy_net.load_state_dict(torch.load(f'../models/model_{num}_policy_net.pth', map_location=device))
policy_net.eval()

def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    game_length = 0
    total_reward = 0
    
    if 'game_over' in info and info['game_over']:
        # handle immediate termination
        reward = torch.tensor([info['reward']], device=device)
        total_reward += reward.item()
    else:    
        while True:
            action = select_action(state)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            total_reward += reward.item()
            game_length += 1
            
            done = terminated or truncated
            if done:
                break
            
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
    cumulative_rewards.append(total_reward)
    game_lengths.append(game_length)
    
    if (i_episode + 1) % 100 == 0:
        print(f"Episode {i_episode + 1}/{num_episodes} - Total Reward: ${sum(cumulative_rewards):.2f}")

average_reward = np.mean(cumulative_rewards)
average_game_length = np.mean(game_lengths)
total_cumulative_reward = np.sum(cumulative_rewards)

print(f"\nEvaluation on {num_episodes} games with ${starting_balance:.2f} starting balance and ${starting_bet:.2f} bets:")
print(f"Total Cumulative Reward: ${total_cumulative_reward:.2f}")
print(f"Average Reward per Game: ${average_reward:.2f}")
print(f"Average Game Length: {average_game_length}")
print(f"Final Balance: ${env.manager.player.balance:.2f}")
