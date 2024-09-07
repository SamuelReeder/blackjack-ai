import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 21
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, n_actions)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# for 18
# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(n_observations, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, n_actions)
        
#         # Initialize weights
#         self.apply(self._init_weights)
    
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#             module.bias.data.fill_(0.01)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return self.fc4(x)

# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         n_dim = 16
#         self.layer1 = nn.Linear(n_observations, n_dim)        
#         self.layer2 = nn.Linear(n_dim, n_dim)        
#         self.layer3 = nn.Linear(n_dim, n_dim)      
#         self.layer4 = nn.Linear(n_dim, n_dim)  
#         self.layer5 = nn.Linear(n_dim, n_actions)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = F.relu(self.layer3(x))
#         x = F.relu(self.layer4(x))
#         return self.layer5(x) 

# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Linear(n_observations, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         self.advantage = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_actions)
#         )
#         self.value = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         x = self.feature(x)
#         advantage = self.advantage(x)
#         value = self.value(x)
#         return value + advantage - advantage.mean()