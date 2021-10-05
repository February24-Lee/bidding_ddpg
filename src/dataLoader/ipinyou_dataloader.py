from .data_loader_base import DataLoaderBase
import sys
# data_path: load features
# om: With or Without market model
# random_om: With random market distribution / With estimated market distribution
#
# mode = 1: including ipinyou market price
# mode = 2: without ipinyou market price
#

class IPinyouDataLoader(DataLoaderBase):
    def __init__(self, 
                data_path : str,
                market_path : str,
                camp : str = '2997',
                file : str ='train.ctr.txt',
                mode : int =2, 
                om : bool =False,
                random_om : bool = False)  -> None :
        '''
        input params:
            - data_path : str
            where ipinyoudataset folder root
            - market_path : str,
            TODO 이친구는 뭐지? 
            - camp : str = '2997',
            - file : str ='train.ctr.txt',
            file의 형태는 기존 bidding 기록을 embedding화여, ctr을 포함한 것.
                col[0] : click
                col[1] : winning price
                col[2] : CTR
                col[3:22] : features. 
            - mode : int =2, 
                mode 2 : pctr
            - om : bool 
            - random_om : bool = False
        '''
        self.data_path = '{}{}/{}'.format(data_path, camp, file)
        self.market_path = market_path
        self.data_source = None
        self.om = om
        self.random_om = random_om
        self.market_source = None
        self.mode = mode
        self.reset()
        self.market = None

    def reset(self) -> None:
        '''
        bid file, market price open
        '''
        self.data_source = open(self.data_path, 'r')
        if self.om:
            if not self.random_om:
                self.market_source = open(self.market_path, 'r')

    def get_next(self, mode : int = None) -> dict:
        '''
        다음의 데이터 laod
        '''
        try:
            raw_bid = next(self.data_source)
        except StopIteration:
            # restart 
            print('data done')
            self.reset()
            raw_bid = next(self.data_source)


        bid = self._construct_bid(raw_bid, self.market, mode)
        return bid

    def get_dataset_length(self) -> int:
        self.reset()
        length = 0
        for _ in self.data_source:
            length += 1
        self.reset()
        return length

    @staticmethod
    def _construct_bid(raw_bid : str,
                        market : list,
                        mode : int) -> dict:
        '''
        str상태의 bid을 받아서,
        dict['click', 'pctr', 'bid'] 형태로 return
        TODO market은 왜 있는지 모르겠음.
        '''
        bid = {}
        raw_bid = raw_bid.strip().split()
        
        bid['click'] = float(raw_bid[0])
        bid['market_price'] = float(raw_bid[1])
        bid['pctr'] = float(raw_bid[2])
        
        #try:
        #    bid['bid'] = list(map(int, raw_bid[3:23]))
        #except ValueError:
        #    convert_float = list(map(float, raw_bid[3:23]))
        #    bid['bid'] = list(map(int, convert_float))
        if market is not None:
            bid['market_price'] = list(map(float, market.strip().split()))
        return bid

    def close(self) -> None:
        self.data_source.close()
        if self.market_source is not None:
            self.market_source.close()