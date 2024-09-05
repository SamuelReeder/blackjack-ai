import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        n_dim = 32
        self.layer1 = nn.Linear(n_observations, n_dim)        
        self.layer2 = nn.Linear(n_dim, n_dim)        
        self.layer3 = nn.Linear(n_dim, n_dim)        
        self.layer4 = nn.Linear(n_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x) 