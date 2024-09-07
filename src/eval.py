import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
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
    print("usage: python src/eval.py <model_number> <optional:for interactive, y or n>")
    sys.exit(1)
num = sys.argv[1]

try:
    interactive = sys.argv[2].lower() == 'y'
except IndexError:
    interactive = False


num_episodes = 1000 
cumulative_rewards = []
game_lengths = []
win_count = 0
loss_count = 0
push_count = 0

env = BlackjackEnv()
starting_balance, starting_bet = env.balance, env.bet
n_actions = env.action_space.n
n_observations = env.state_size

policy_net = DQN(n_observations, n_actions).to(device)

policy_net.load_state_dict(torch.load(f'./models/model_{num}_policy_net.pth', map_location=device))
policy_net.eval()

def plot_metrics(filename="eval_metrics.png"):
    plt.figure(1)
    balances_t = torch.tensor(env.balances, dtype=torch.float)
    
    plt.title('Evaluation Performance')
    
    plt.xlabel('Episode')
    plt.ylabel('Balance')
    
    plt.plot(balances_t.numpy(), linestyle='-', color='blue') 
    
    plt.ylim([balances_t.min().item(), balances_t.max().item()])
    
    plt.savefig(filename)
    
    print(f"Plot saved as {filename}")
    
    
def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    game_length = 0
    total_reward = 0
    
    if interactive:
        print(f"\n=== Game {i_episode + 1} ===")
        print(f"Player Balance: ${env.manager.player.balance:.2f}")
        print(f"Player Bet: ${env.manager.player.hands[0].bet:.2f}")
        print(f"Dealer's Visible Card: {env.manager.dealer.hand.calculate_value(hide_dealer=True)}")
        print(f"Player's Hand: {env.manager.player.hands[0].cards} (Value: {env.manager.player.hands[0].calculate_value()})")
        print('State:', state)
    
    if 'game_over' in info and info['game_over']:
        # handle immediate termination
        reward = torch.tensor([info['reward']], device=device)
        total_reward += reward.item()
        
        # print(f"Game {i_episode + 1} - Immediate Termination - Reward: ${reward.item():.2f}") if interactive else None
    else:    
        while True:
            action = select_action(state)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            total_reward += reward.item()
            
            if interactive:
                print(f"\n--- Player's Turn ---")
                print(f"Action: {action.item()}")
                print(f"Player's Hand: {env.manager.player.hands[0].cards} (Value: {env.manager.player.hands[0].calculate_value()})")
                print(f"Dealer's Visible Card: {env.manager.dealer.hand.calculate_value(hide_dealer=True)}")
                print('State:', observation)
            
            done = terminated or truncated
            if done:
                # print(f"Game {i_episode + 1} - Total Reward: ${total_reward:.2f}")
                # input("Press Enter to continue... ") if interactive else None
                break
            
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
    cumulative_rewards.append(total_reward)
    game_lengths.append(game_length)
    
    if total_reward > 0:
        win_count += 1
    elif total_reward < 0:
        loss_count += 1
    else:
        push_count += 1
        
        
    if (i_episode + 1) % 100 == 0:
        print(f"Episode {i_episode + 1}/{num_episodes} - Total Reward: ${sum(cumulative_rewards):.2f}")
    
    if interactive:
        print(f"\n--- END ---")
        print(f"Dealer's Final Hand: {env.manager.dealer.hand.cards} (Value: {env.manager.dealer.hand.calculate_value()})")
        print(f"Game {i_episode + 1} - Total Reward: ${total_reward:.2f}")
        input("Press Enter to continue... ")

average_reward = np.mean(cumulative_rewards)
average_game_length = np.mean(game_lengths)
total_cumulative_reward = np.sum(cumulative_rewards)
win_rate = (win_count / num_episodes) * 100
loss_rate = (loss_count / num_episodes) * 100
push_rate = (push_count / num_episodes) * 100

print(f"\nEvaluation on {num_episodes} games with ${starting_balance:.2f} starting balance and ${starting_bet:.2f} bets:")
print(f"Total Cumulative Reward: ${total_cumulative_reward:.2f}")
print(f"Average Reward per Game: ${average_reward:.2f}")
print(f"Average Game Length: {average_game_length}")
print(f"Win Count: {win_count} ({win_rate:.2f}% win rate)")
print(f"Loss Count: {loss_count} ({loss_rate:.2f}% loss rate)")
print(f"Push Count: {push_count} ({push_rate:.2f}% push rate)")
print(f"Final Balance: ${env.manager.player.balance:.2f}")

plot_metrics(filename=f'eval_metric_{num}.png')
