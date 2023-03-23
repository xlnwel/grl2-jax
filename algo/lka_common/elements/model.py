import numpy as np
from copy import deepcopy
import chex

from algo.ma_common.elements.model import *

LOOKAHEAD = 'lookahead'
PREV_PARAMS = 'prev_params'
PREV_LKA_PARAMS = 'prev_lka_params'


class LKAModelBase(MAModelBase):
    def add_attributes(self):
        super().add_attributes()
        self.lookahead_params = AttrDict()
        self.prev_params = AttrDict()
        self.prev_lka_params = AttrDict()

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
        self.lookahead_params = deepcopy(self.params)
        # self.lookahead_params.policies = [p.copy() for p in self.lookahead_params.policies]
        # self.lookahead_params.vs = [v.copy() for v in self.lookahead_params.vs]
        for p in self.lookahead_params.policies:
            p[LOOKAHEAD] = True

    def joint_policy(self, params, rng, data):
        params, _ = pop_lookahead(params)
        return super().joint_policy(params, rng, data)

    def set_params(self, params):
        self.prev_params = self.params.copy()
        self.prev_params[PREV_PARAMS] = True
        self.params = params
    
    def set_lka_params(self, params):
        self.prev_lka_params = self.lookahead_params.copy()
        self.prev_lka_params[PREV_LKA_PARAMS] = True
        self.lookahead_params = params

    def swap_params(self):
        self.prev_params, self.params = self.params, self.prev_params

    def swap_lka_params(self):
        self.prev_lka_params, self.lookahead_params = self.lookahead_params, self.prev_lka_params

    def check_current_params(self):
        assert PREV_PARAMS not in self.params, list(self.params)
        assert PREV_PARAMS in self.prev_params, list(self.prev_params)
    
    def check_current_lka_params(self):
        assert PREV_LKA_PARAMS not in self.lookahead_params, list(self.lookahead_params)
        assert PREV_LKA_PARAMS in self.prev_lka_params, list(self.prev_lka_params)


def pop_lookahead(policies):
    policies = [p.copy() for p in policies]
    lka = [p.pop(LOOKAHEAD, False) for p in policies]
    return policies, lka
