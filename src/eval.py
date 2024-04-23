import matplotlib
import matplotlib.pyplot as plt
import torch
from environment import BlackjackEnv, ReplayMemory
from network import DQN
import numpy as np

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BlackjackEnv()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

# Assuming policy_net is your model and has the same architecture as the saved one
policy_net.load_state_dict(torch.load('../models/model_5_policy_net.pth'))

policy_net.eval()  # Set the network to evaluation mode

balances = []

# outcomes = {wins: 0, losses: 0, ties: 0}

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(balances, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('what is happening...')
    plt.xlabel('episode')
    plt.ylabel('balance')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# def plot_outcomes(show_result=False):
#     plt.figure(2)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Balance')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     # if len(durations_t) >= 100:
#     #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#     #     means = torch.cat((torch.zeros(99), means))
#     #     plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())

# num_test_episodes = 1000

# for i_episode in range(num_test_episodes):
#     state = env.reset()
#     state = torch.tensor([state], dtype=torch.float32)  # Assuming you need to format the state like this
#     for t in count():
#         # Select an action
#         action = select_action(state)  # Your function to choose the best action given the current state
#         next_state, reward, done, _ = env.step(action.item())
#         balances.append(env.game.players[0].balance)
        
#         if done:
#             if reward > 0:
#                 outcomes[wins] += 1
#             elif reward < 0:
#                 outcomes[losses] += 1
#             else:
#                 outcomes[ties] += 1
        
#         state = next_state
#         if done:
#             print(f"Episode finished after {t+1} timesteps")
#             plot_durations()
#             break

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
