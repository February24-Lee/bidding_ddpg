from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import fanin_init

class Actor(nn.Module):
    def __init__(self,
                dim_state : int = None,
                dim_action : int = None,
                hidden1 : int = 16,
                hidden2 : int = 16,
                init_w : float = 3e-3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_state, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, dim_action)
        self.init_weights(init_w)
        
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.sigmoid(self.fc3(x))
        return out
    

class Critic(nn.Module):
    def __init__(self,
                dim_states : int ,
                dim_action : int,
                hidden1 : int = 16,
                hidden2 : int = 16,
                init_w=3e-3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + dim_action, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, x:List= None) -> float:
        state, action = x
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat([x, action], 1)))
        out = self.fc3(x)
        return out