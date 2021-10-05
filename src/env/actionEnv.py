from typing import List
import gym
from gym.spaces import box
import numpy as np

from ..dataLoader.ipinyou_dataloader import IPinyouDataLoader

class ActionEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self,
                dataloader          : IPinyouDataLoader = None,
                episode_maxlen      : int               = 1000) -> None:
        super(ActionEnv, self).__init__()
        self.dataloader             = dataloader
        self.episode_maxlen         = episode_maxlen
        self.num_action             = 0
        self.episode_idx            = 0
        self.now_action             = {}
        self.reset()
        
    def reset(self):
        self.episode_idx    += 1
        self.num_action     = 1
        self.now_action     = self.dataloader.get_next()
        return self.now_action
        
    def step(self, actions : np.ndarray) -> List:
        agent_num       = len(actions)
        winner_idx      = np.argmax(actions)
        market_price    = actions[np.argsort(actions)[-2]] # --- second 
        winner_click    = self.now_action['click']
        winner_pctr     = self.now_action['pctr']
        
        reward = np.zeros((agent_num))
        reward[winner_idx] = 1
        info = {'market_price' : market_price,
                'click'        : winner_click,
                'pctr'         : winner_pctr}
        
        self.num_action += 1
        self.now_action = self.dataloader.get_next()
        if self.num_action >= self.episode_maxlen :
            terminal = True
        else :
            terminal = False
        
        return self.now_action, reward, terminal, info
        
        