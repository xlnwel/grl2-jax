from utility.utils import dict2AttrDict


class PayoffTable:
    def __init__(self, config):
        self.config = dict2AttrDict(config)
    
    