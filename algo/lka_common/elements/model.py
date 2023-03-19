import numpy as np

from algo.ma_common.elements.model import *

LOOKAHEAD = 'lookahead'


class LKAModelBase(MAModelBase):
    def add_attributes(self):
        super().add_attributes()
        self.lookahead_params = AttrDict()

    def switch_params(self, lookahead, aids=None):
        if aids is None:
            aids = np.arange(self.n_agents)
        for i in aids:
            self.params.policies[i], self.lookahead_params.policies[i] = \
                self.lookahead_params.policies[i], self.params.policies[i]
        self.check_params(lookahead, aids)

    def check_params(self, lookahead, aids=None):
        if aids is None:
            aids = np.arange(self.n_agents)
        for i in aids:
            assert self.params.policies[i][LOOKAHEAD] == lookahead, (self.params.policies[i][LOOKAHEAD], lookahead)
            assert self.lookahead_params.policies[i][LOOKAHEAD] == 1-lookahead, (self.lookahead_params.policies[i][LOOKAHEAD], lookahead)

    def sync_lookahead_params(self):
        self.lookahead_params = self.params.copy()
        self.lookahead_params.policies = [p.copy() for p in self.lookahead_params.policies]
        for p in self.lookahead_params.policies:
            p[LOOKAHEAD] = True
