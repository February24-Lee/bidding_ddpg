from .base_agent import BaseAgent
import numpy as np

class McpcAgent(BaseAgent):
    def __init__(self,
                camp_info       : dict = None,
                max_bid_price   : int = None,
                budget          : int = None) -> None:
        """Linear Agent 생성

        Args:
            load_info (str, optional): [이전 기록이 있으면 불러오기. 형태는 pickle]. Defaults to None.
        """
        super().__init__()
        
        self.origin_budget      = budget
        self.remained_budget    = budget
        self.num_win            = 0
        self.num_attend_bid     = 0
        self.num_click          = 0
        self.list_pctr          = []
        self.camp_info          = camp_info
        
        if max_bid_price is not None:
            self.max_bid_price = max_bid_price
    
    @property
    def max_bid_price(self):
        return self._max_bid_price
    
    @max_bid_price.setter
    def max_bid_price(self, price):
        self._max_bid_price = price 
        
    def loadCampInfo(self) -> None:
        self.train_cpm          = self.camp_info["cost_train"] / self.camp_info["imp_train"]
        self.train_cpc          = self.camp_info['cost_train'] / self.camp_info['clk_train']
        
        self.test_cpc           = self.camp_info["cost_test"] / self.camp_info["clk_test"]
        self.test_cpm           = self.camp_info['cost_test'] / self.camp_info['imp_test']
    
    def action(self, observed_state, is_training     : bool = True) -> int:
        if is_training:
            cpc = self.train_cpc
        else :
            cpc = self.test_cpc
        action = min(observed_state * cpc, self.max_bid_price)
        action = min(self.remained_budget, action)
        return action
        
        
        
        
    