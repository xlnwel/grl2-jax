import jax

from core.typing import dict2AttrDict
from tools.timer import timeit
from algo.lka_common.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def train(self, step, **kwargs):
        self.lookahead_train(**kwargs)
        return super().train(step, **kwargs)

    @timeit
    def _after_train(self, stats):
        self.rng, rng = jax.random.split(self.rng, 2)
        mu_keys = self.model.policy_dist_stats('mu')
        data = dict2AttrDict({k: v 
            for k, v in self.training_data.items() 
            if k in ['obs', 'global_state', *mu_keys]
        })
        if data is None:
            return stats
        
        pi_dist = self.model.joint_policy(
            self.model.theta.policies, rng, data
        )
        
        mu_dist = self.model.joint_policy(
            self.prev_params.policies, rng, data
        )
        kl_mu_pi = mu_dist.kl_divergence(pi_dist)
        stats.kl_mu_pi = kl_mu_pi

        lka_dist = self.model.joint_policy(
            self.model.lookahead_params.policies, rng, data
        )
        kl_lka_pi = lka_dist.kl_divergence(pi_dist)
        stats.kl_lka_pi = kl_lka_pi
        stats.kl_mu_lka_diff = kl_mu_pi - kl_lka_pi

        mix_policies = [self.prev_params.policies[0]]
        mix_policies += self.model.lookahead_params.policies[1:]
        mix_dist = self.model.joint_policy(
            mix_policies, rng, data
        )
        kl_mix_pi = mix_dist.kl_divergence(pi_dist)
        stats.kl_mix_pi = kl_mix_pi
        stats.kl_mu_mix_diff = kl_mu_pi - kl_mix_pi

        return stats
