import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np


# class DQN(nn.Module):
#     def __init__(self, state_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.action_head = nn.Linear(64, 5)  # 5 discrete actions
#         # self.bet_amount_head = nn.Linear(64, 1)  # Continuous bet amount

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         action_probs = F.softmax(self.action_head(x), dim=-1)
#         # bet_amount = torch.sigmoid(self.bet_amount_head(x))  # Assuming normalized bet amount
#         return action_probs  # , bet_amount
    
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # return self.layer3(x)
        return F.softmax(self.layer4(x), dim=-1)
