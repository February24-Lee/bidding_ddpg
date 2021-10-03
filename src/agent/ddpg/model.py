from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import fanin_init

class Actor(nn.Module):
    def __init__(self,
                dim_state   : int = None,
                dim_action  : int = None,
                num_layer   : int = None,
                dim_layer   : int = None,
                init_w      : float = 3e-3) -> None:
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(dim_state, dim_layer))
        for _ in range(num_layer):
            self.module_list.append(nn.Linear(dim_layer, dim_layer))
        self.module_list.append(nn.Linear(dim_layer, dim_action))
        self.init_weights(init_w)
        
    def init_weights(self, init_w):
        for select_layer in self.module_list[:-1]:
            select_layer.weight.data = fanin_init(select_layer.weight.data.size())

        self.module_list[-1].weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        for layer in self.module_list[:-1]:
            x = F.relu(layer(x))
        out = F.sigmoid(self.module_list[-1](x))
        return out
    

class Critic(nn.Module):
    def __init__(self,
                dim_states  : int ,
                dim_action  : int,
                num_layer   : int = None,
                dim_layer   : int = None,
                init_w=3e-3) -> None:
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(dim_states, dim_layer))
        #self.module_list.append(nn.Linear(dim_layer + dim_action, dim_layer))
        for _ in range(num_layer-1):
            self.module_list.append(nn.Linear(dim_layer, dim_layer))
        self.module_list.append(nn.Linear(dim_layer, 1))
        
    def init_weights(self, init_w):
        for layer in self.module_list[:-1]:
            layer.weight.data = fanin_init(layer.weight.data.size())
        self.module_list[-1].data.uniform_(-init_w, init_w)
        
    def forward(self, x:List= None) -> float:
        state, action = x
        #x = F.relu(self.module_list[0](state))
        x = F.relu(self.module_list[0](torch.cat([state, action], 1)))
        for layer in self.module_list[1:-1]:
            x = F.relu(layer(x))
        out = self.module_list[-1](x)
        return out