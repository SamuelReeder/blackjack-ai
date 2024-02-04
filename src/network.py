import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np


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
        return F.softmax(self.layer4(x), dim=-1)

# class DQN(nn.Module):

#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.bn1 = nn.BatchNorm1d(128)  # Batch normalization layer after layer1
#         self.dropout1 = nn.Dropout(p=0.2)  # Dropout layer after activation with a dropout rate of 20%
        
#         self.layer2 = nn.Linear(128, 128)
#         self.bn2 = nn.BatchNorm1d(128)  # Batch normalization layer after layer2
#         self.dropout2 = nn.Dropout(p=0.2)  # Dropout layer after activation with a dropout rate of 20%
        
#         self.layer3 = nn.Linear(128, 128)
#         self.bn3 = nn.BatchNorm1d(128)  # Batch normalization layer after layer3
#         self.dropout3 = nn.Dropout(p=0.2)  # Dropout layer after activation with a dropout rate of 20%
        
#         self.layer4 = nn.Linear(128, n_actions)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.layer1(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.bn2(self.layer2(x)))
#         x = self.dropout2(x)
#         x = F.relu(self.bn3(self.layer3(x)))
#         x = self.dropout3(x)
#         return F.softmax(self.layer4(x), dim=-1)