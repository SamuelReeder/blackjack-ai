import torch
from environment import BlackjackEnv

env = BlackjackEnv()



# Assuming policy_net is your model and has the same architecture as the saved one
policy_net.load_state_dict(torch.load('./models/model_0_policy_net.pth'))

policy_net.eval()  # Set the network to evaluation mode

balances = []

outcomes = {wins: 0, losses: 0, ties: 0}

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(balances, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Balance')
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

def plot_outcomes(show_result=False):
    plt.figure(2)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Balance')
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

num_test_episodes = 1000

for i_episode in range(num_test_episodes):
    state = env.reset()
    state = torch.tensor([state], dtype=torch.float32)  # Assuming you need to format the state like this
    for t in count():
        # Select an action
        action = select_action(state)  # Your function to choose the best action given the current state
        next_state, reward, done, _ = env.step(action.item())
        balances.append(env.game.players[0].balance)
        
        if done:
            if reward > 0:
                outcomes[wins] += 1
            elif reward < 0:
                outcomes[losses] += 1
            else:
                outcomes[ties] += 1
        
        state = next_state
        if done:
            print(f"Episode finished after {t+1} timesteps")
            plot_durations()
            break
        
print(outcomes)
