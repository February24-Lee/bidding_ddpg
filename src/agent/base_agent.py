class BaseAgent(object):
    def action(self, observed_state):
        NotImplemented
        
    def train(self):
        NotImplemented
        
    def update_result(self, 
                    click : int = None,
                    is_win : bool = None,
                    market_price:int = None,
                    pctr : float = None):
        self.num_attend_bid += 1
        if is_win:
            self.num_win += 1
            self.num_click += click
            self.list_pctr.append(pctr)
            self.remained_budget -= market_price
            
    def reset(self) -> None:
        self.remained_budget    = self.origin_budget
        self.num_win            = 0
        self.num_attend_bid     = 0
        self.num_click          = 0
        self.list_pctr          = []