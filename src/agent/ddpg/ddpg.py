import torch
import torch.optim as optim
import torch.nn as nn

import copy
import numpy as np
import wandb

from ..base_agent import BaseAgent
from .model import Critic, Actor
from .utils import soft_update
from .randomProcess import OrnsteinUhlenbeckProcess
from .memory import SequentialMemory


class DDPGAgent(BaseAgent):
    def __init__(self,
                dim_state           : int,
                dim_action          : int,
                actor_optim_lr      : float = 0.001,
                critic_optim_lr     : float = 0.001,
                ou_theta            : float = None,
                ou_mu               : float = None,
                ou_sigma            : float = None,
                memory_size         : int   = None,
                window_length       : int   = 1,
                batch_size          : int   = 32,
                soft_copy_tau       : float = None,
                discount            : float = None,
                epsilon             : float = None,
                max_bid_price       : int   = None,
                budget              : float = None,
                logger                      = None,
                num_actor_layer     : int   = None,
                dim_actor_layer     : int   = None,
                num_critic_layer    : int   = None,
                dim_critic_layer    : int   = None) -> None:
        super().__init__()
        
        self.actor              = Actor(dim_state   = dim_state, 
                                        dim_action  = dim_action,
                                        num_layer   = num_actor_layer,
                                        dim_layer   = dim_actor_layer)
        
        self.actor_target       = copy.deepcopy(self.actor)
        self.actor_optim        = optim.Adam(self.actor.parameters(), lr = actor_optim_lr)
        
        self.critic             = Critic(dim_states     = dim_state,
                                        dim_action      = dim_action,
                                        num_layer       = num_critic_layer,
                                        dim_layer       = dim_critic_layer)
        self.critic_target      = copy.deepcopy(self.critic)
        self.critic_optim       = optim.Adam(self.critic.parameters(), lr = critic_optim_lr)
        self.random_process     = OrnsteinUhlenbeckProcess(size=dim_action,
                                                    theta=ou_theta,
                                                    mu = ou_mu,
                                                    sigma=ou_sigma)
        self.memory             = SequentialMemory(limit=memory_size,
                                                    window_length = window_length) 
        
        self.batch_size         = batch_size
        self.soft_copy_tau      = soft_copy_tau
        self.discount           = discount
        self.epsilon            = epsilon
        self.epsilon_copy       = epsilon
        self.is_training        = True
        self.max_bid_price      = max_bid_price
        self.origin_budget      = budget
        self.remained_budget    = budget
        self.before_remained_budget = budget
        self.num_attend_bid     = 0
        self.num_win            = 0
        self.num_click          = 0
        self.list_pctr          = []
        
        self.device             = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.criterion          = nn.MSELoss()
        self.actor              = self.actor.to(self.device)
        self.actor_target       = self.actor_target.to(self.device)
        self.critic             = self.critic.to(self.device)
        self.critic_target      = self.critic_target.to(self.device)
        self.logger             = logger
        
        
    @property
    def is_training(self):
        return self._is_training
    
    @is_training.setter
    def is_training(self, x):
        self._is_training = x
        
    def eval(self) -> None:
        self.actor              = self.actor.eval()
        self.actor_target       = self.actor_target.eval()
        self.critic             = self.critic.eval()
        self.critic_target      = self.critic_target.eval()
    
    def update_policy(self, step: int = None) -> None:
        # sample batch
        state0_batch, action_batch, reward_batch, \
            state1_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        
        next_q_values = self.critic_target([torch.from_numpy(state1_batch).to(self.device),
                                            self.actor_target(torch.from_numpy(state1_batch).to(self.device))])
        target_q_values = torch.from_numpy(reward_batch).to(self.device) + \
            self.discount * torch.from_numpy(terminal_batch.astype(np.float)).to(self.device) * next_q_values
        
        # --- training critic
        self.critic_optim.zero_grad()
        hat_q_batch = self.critic([torch.from_numpy(state0_batch).to(self.device), torch.from_numpy(action_batch).to(self.device)])
        critic_loss = self.criterion(hat_q_batch, target_q_values)
        
        # --- log 
        wandb.log({'critic_loss' : critic_loss.item(), 'step':step})
        if self.logger is not None:
            #self.logger.add_scalar('critic_loss', critic_loss.item() ,global_step =step)
            self.logger.log({'critic_loss' : critic_loss.item()} ,global_step = step)
        critic_loss.backward()
        self.critic_optim.step()
        
        # --- training actor
        self.actor_optim.zero_grad()
        policy_loss = -self.critic([torch.from_numpy(state0_batch).to(self.device), self.actor(torch.from_numpy(state0_batch).to(self.device))])
        policy_loss = policy_loss.mean()
        wandb.log({'policy_loss' : policy_loss.item(), 'step':step})
        if self.logger is not None:
            self.logger.log({'policy_loss' : policy_loss.item()},global_step =step)
        policy_loss.backward()
        self.actor_optim.step()
        
        soft_update(self.actor_target, self.actor, self.soft_copy_tau)
        soft_update(self.critic_target, self.critic, self.soft_copy_tau)
        
    
    def fill_memory(self, state0, action, reward, terminal):
        if self.is_training:
            self.memory.append(state0, action, reward, terminal)
        
        
    def action(self,
                obs                 : np.ndarray    = None,
                remained_opport     : int           = None,
                is_decay_epsilon    : bool          = True) -> np.ndarray:
        '''
        state :
            - pCTR
            - Remaining budget divided by the initiial budget, 
            - the number of regulation opportunitites left (0 < x < 1)
            - the budget consumption rate
            - auction win rate 
            - 
        '''
        input_x = np.array([obs,
                            self.remained_budget/self.origin_budget,
                            remained_opport,
                            (self.before_remained_budget/self.remained_budget)/(self.before_remained_budget+1e-5),
                            self.num_win/(self.num_attend_bid+1e-5),
                            self.num_click], dtype=np.float32)
        actor_action = self.actor(torch.from_numpy(input_x).to(self.device)).cpu().detach().numpy()
        actor_action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        actor_action = np.clip(actor_action, 0., 1.)
        
        actor_action = self.max_bid_price * actor_action
        actor_action = actor_action[0]
        actor_action = min(self.remained_budget, actor_action).astype(np.float32)
        if is_decay_epsilon:
            self.epsilon = 1/self.epsilon
            
        return input_x, actor_action
    
    
    def random_action(self, obs, remained_opport):
        input_x = np.array([obs,
                            self.remained_budget/self.origin_budget,
                            remained_opport,
                            (self.before_remained_budget/self.remained_budget)/(self.before_remained_budget+1e-5),
                            self.num_win/(self.num_attend_bid+1e-5),
                            self.num_click], dtype=np.float32)
        random_action = min(np.random.rand(1) * self.max_bid_price, self.remained_budget).astype(np.float32)
        return input_x, random_action[0]
        
            
            
    
        
        
        
        
        