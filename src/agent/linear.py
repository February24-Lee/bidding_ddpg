from .base_agent import BaseAgent
import pickle
import numpy as np

class LinearAgent(BaseAgent):
    def __init__(self,
                load_info       : str = None,
                camp_info       : dict = None,
                max_bid_price   : int = None,
                budget          : int = None) -> None:
        """Linear Agent 생성

        Args:
            load_info (str, optional): [이전 기록이 있으면 불러오기. 형태는 pickle]. Defaults to None.
        """
        super().__init__()
        
        self.b0                 = 0
        self.origin_budget      = budget
        self.remained_budget    = budget
        self.num_win            = 0
        self.num_attend_bid     = 0
        self.num_click          = 0
        self.list_pctr          = []
        
        if load_info is not None:
            with open(load_info, 'rb') as f:
                args = pickle.load(f)
            self.b0 =args["b0"]
            
        if camp_info is not None:
            self.loadCampInfo(camp_info)
            
        if max_bid_price is not None:
            self.max_bid_price = max_bid_price
    
    @property
    def max_bid_price(self):
        return self._max_bid_price
    
    @max_bid_price.setter
    def max_bid_price(self, price):
        self._max_bid_price = price 

    @property
    def b0(self):
        return self._b0
    
    @b0.setter
    def b0(self, new_bo):
        self._b0 = new_bo
    
    def loadCampInfo(self, camp_info:dict) -> None:
        """ Camp의 info을 가져와 cpm 및 theta_avg 계산

        Args:
            camp_info (dict): 
        """
        self.train_cpm          = camp_info["cost_train"]/camp_info["imp_train"]
        self.train_theta_avg    = camp_info['clk_train']/camp_info['imp_train']
        self.test_cpm           = camp_info["cost_test"]/camp_info["imp_test"]
        self.test_theta_avg     = camp_info['clk_test']/camp_info['imp_test']
        
    def action(self,
                observed_state  : dict = None,
                is_training     : bool = True) -> int:
        if is_training:
            theta_avg = self.train_theta_avg
        else:
            theta_avg = self.test_theta_avg
            
        action = min(int(observed_state * self.b0 / theta_avg), self.max_bid_price)
        action = min(self.remained_budget, action)
        return action

        
    
    
