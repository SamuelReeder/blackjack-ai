import gymnayzium
env = BlackjackEnv()  # Assuming BlackjackEnv is your custom environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = DQNNetwork(state_size, action_size)


