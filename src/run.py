import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import blackjack
from model import DQN
import math
import sys
import random
from collections import namedtuple, deque
from itertools import count
from environment import ReplayMemory, BlackjackEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


if len(sys.argv) < 2:
    print("Please provide a number to save the model")
    sys.exit(1)
num = sys.argv[1]

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# hyperparameters
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 3e-4
REPORT_INTERVAL = 500
num_episodes = 10000

env = BlackjackEnv()
n_actions = env.action_space.n
n_observations = env.state_size

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(5000)

steps_done = 0

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_metrics(show_result=False, filename='training_progress.png'):
    plt.figure(1)
    balances_t = torch.tensor(env.balances, dtype=torch.float)
    
    if show_result:
        plt.title('Final Result')
        filename = 'final_result.png'
    else:
        plt.clf()
        plt.title('Training Progress...')
    
    plt.xlabel('Episode')
    plt.ylabel('Balance')
    
    plt.plot(balances_t.numpy(), linestyle='-', color='blue')  # Line plot
    
    plt.ylim([balances_t.min().item(), balances_t.max().item()])
    
    plt.savefig(filename)
    
    print(f"Plot saved as {filename}")


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q values for current states
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Q values for next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute loss and optimize
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def run(random_play=False):
    total_reward = 0
    rewards = []
    thousand_episode_rewards = 0
    
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if 'game_over' in info and info['game_over']:
            # Handle immediate termination
            reward = torch.tensor([info['reward']], device=device)
            total_reward += reward.item()
            rewards.append(reward.item())
        else:
            for t in count():
                action = select_action(state) if not random_play else torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                total_reward += reward.item()
                rewards.append(reward.item())
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                memory.push(state, action, next_state, reward)
                state = next_state

                optimize_model()

                # soft update of target network
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
        
        # env.balances.append(total_reward)
        if i_episode % REPORT_INTERVAL == 0:
            sum_rewards = sum(rewards)
            print(f"Episode {i_episode}/{num_episodes} - Total Reward: {total_reward} - Average Reward: {sum_rewards / REPORT_INTERVAL}")
            thousand_episode_rewards += sum_rewards
            rewards = []
            if i_episode % 1000 == 0:
                print(f"\nAverage Reward for last 1000 episodes: {thousand_episode_rewards / 1000}\n")
                thousand_episode_rewards = 0
run()

torch.save(policy_net.state_dict(), f'./models/model_{num}_policy_net.pth')
torch.save(target_net.state_dict(), f'./models/model_{num}_target_net.pth')
torch.save({
    'episode': num_episodes,
    'model_state_dict': policy_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f'./models/model_{num}_checkpoint.pth')
print(f'Model saved as model_{num}.pth')

plot_metrics(show_result=True)
# plt.ioff()
# plt.show()
