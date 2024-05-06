import matplotlib
import matplotlib.pyplot as plt
import torch
from environment import BlackjackEnv, ReplayMemory
from network import DQN
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Please provide a number to save the model")
    sys.exit(1)
num = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BlackjackEnv()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

# Assuming policy_net is your model and has the same architecture as the saved one
policy_net.load_state_dict(torch.load(f'../models/model_{str(num)}_policy_net.pth'))

policy_net.eval()  # Set the network to evaluation mode

balances = []



def test(num_test_episodes, policy_net, env, device):
    total_rewards = []  # To store total rewards for each episode

    for i_episode in range(num_test_episodes):
        # Initialize the environment and state
        state, info = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32)
        episode_reward = 0  # Accumulates rewards for this episode

        while True:
            # Select action based purely on policy (no randomness)
            with torch.no_grad():  # Important: do not compute gradients
                action = policy_net(state).max(1)[1].view(1, 1)
            
            # Perform action in env
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            balances.append(env.game.players[0].balance)
            
            episode_reward += reward

            if terminated or truncated:
                plot_durations()
                break  # Exit loop if the episode ended

            # Move to the next state
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
            state = next_state

        total_rewards.append(episode_reward)  # Store the total reward for this episode

    # Calculate and print the average reward
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {num_test_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

print("Testing the model...")
test(10000, policy_net, env, device)

# print(outcomes)
