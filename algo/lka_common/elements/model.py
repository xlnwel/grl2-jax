import numpy as np
from copy import deepcopy
from jax import random
import optax

from core.log import do_logging
from tools.timer import timeit
from jax_tools import jax_dist, jax_div
from algo.ma_common.elements.model import *

LOOKAHEAD = 'lookahead'
PREV_PARAMS = 'prev_params'
PREV_LKA_PARAMS = 'prev_lka_params'


class LKAModelBase(MAModelBase):
    def add_attributes(self):
        super().add_attributes()
        self.rng, self.pd_rng = random.split(self.rng)
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

    @timeit
    @partial(jax.jit, static_argnums=[0, 3])
    def compute_policy_distances(self, data, stats=AttrDict(), eval_lka=True):
        do_logging('compute_policy_distances is traced')
        if self.has_rnn:
            data.state = tree_slice(data.state, indices=0, axis=1)

        self.pd_rng, rng = random.split(self.pd_rng)
        pi_dist = self.joint_policy(
            self.theta.policies, rng, data
        )
        
        mu_dist = self.joint_policy(
            self.prev_params.policies, rng, data
        )
        kl_mu_pi = mu_dist.kl_divergence(pi_dist)
        stats.kl_mu_pi = kl_mu_pi
        js_mu_pi = dist_js(mu_dist, pi_dist)
        stats.js_mu_pi = js_mu_pi

        if eval_lka:
            lka_dist = self.joint_policy(
                self.prev_lka_params.policies, rng, data
            )
            kl_lka_pi = lka_dist.kl_divergence(pi_dist)
            stats.kl_lka_pi = kl_lka_pi
            stats.kl_mu_lka_diff = kl_mu_pi - kl_lka_pi

            mix_policies = [self.prev_params.policies[0]]
            mix_policies += self.prev_lka_params.policies[1:]
            mix_dist = self.joint_policy(
                mix_policies, rng, data
            )
            kl_mix_pi = mix_dist.kl_divergence(pi_dist)
            stats.kl_mix_pi = kl_mix_pi
            stats.kl_mu_mix_diff = kl_mu_pi - kl_mix_pi

            js_lka_pi = dist_js(lka_dist, pi_dist)
            js_mix_pi = dist_js(mix_dist, pi_dist)
            stats.js_lka_pi = js_lka_pi
            stats.js_mix_pi = js_mix_pi
            stats.js_mu_lka_diff = js_mu_pi - js_lka_pi
            stats.js_mu_mix_diff = js_mu_pi - js_mix_pi

        if isinstance(pi_dist, jax_dist.Categorical):
            stats.cos_mu_pi = dist_cos(mu_dist, pi_dist)
            if eval_lka:
                stats.cos_mu_lka = dist_cos(mu_dist, lka_dist)
                stats.cos_lka_pi = dist_cos(lka_dist, pi_dist)
                stats.cos_mix_pi = dist_cos(mix_dist, pi_dist)
                stats.cos_lka_mu_diff = stats.cos_lka_pi - stats.cos_mu_pi
                stats.cos_mix_mu_diff = stats.cos_mix_pi - stats.cos_mu_pi
        else:
            stats.wasserstein_mu_pi = dist_wasserstein(mu_dist, pi_dist)
            if eval_lka:
                stats.wasserstein_mu_lka = dist_wasserstein(mu_dist, lka_dist)
                stats.wasserstein_lka_pi = dist_wasserstein(lka_dist, pi_dist)
                stats.wasserstein_mix_pi = dist_wasserstein(mix_dist, pi_dist)
                stats.wasserstein_lka_mu_diff = stats.wasserstein_lka_pi - stats.wasserstein_mu_pi
                stats.wasserstein_mix_mu_diff = stats.wasserstein_mix_pi - stats.wasserstein_mu_pi

        return stats


def dist_js(d1, d2):
    return jax_div.js_from_distributions(
        **d1.get_stats('p'), **d2.get_stats('q')
    )


def dist_wasserstein(d1, d2):
    return jax_div.wasserstein(
        **d1.get_stats('p'), **d2.get_stats('q')
    )


def dist_cos(d1, d2):
    return optax.cosine_similarity(d1.probs, d2.probs)


def pop_lookahead(policies):
    policies = [p.copy() for p in policies]
    lka = [p.pop(LOOKAHEAD, False) for p in policies]
    return policies, lka
