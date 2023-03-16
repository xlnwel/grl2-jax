import jax

from core.typing import AttrDict, dict2AttrDict
from tools.timer import timeit
from algo.lka_common.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    @timeit
    def _after_train(self, stats: AttrDict):
        self.rng, rng = jax.random.split(self.rng, 2)
        mu_keys = self.model.policy_dist_stats('mu')
        data = dict2AttrDict({k: v 
            for k, v in self.training_data.items() 
            if k in ['obs', 'global_state', *mu_keys]
        })
        if data is None:
            return stats
        pi_params = self.model.joint_policy(
            self.model.theta.policies, rng, data
        )
        pi_dist = self.model.policy_dist(pi_params)

        mu_params = self.model.joint_policy(
            self.prev_params.policies, rng, data
        )
        mu_dist = self.model.policy_dist(mu_params)
        kl_mu_pi = mu_dist.kl_divergence(pi_dist)

        behavior_params = [
            data[mk].reshape(*data[mk].shape[:-2], 1, -1) 
            for mk in mu_keys
        ]
        lka_dist = self.model.policy_dist(behavior_params)
        kl_lka_pi = lka_dist.kl_divergence(pi_dist)

        stats.kl_mu_pi = kl_mu_pi
        stats.kl_lka_pi = kl_lka_pi
        stats.kl_mu_lka_diff = kl_mu_pi - kl_lka_pi

        mix_policies = [self.model.theta.policies[0]]
        mix_policies += self.model.lookahead_params.policies[1:]
        mix_params = self.model.joint_policy(
            mix_policies, rng, data
        )
        mix_dist = self.model.policy_dist(mix_params)
        stats.kl_mix_pi = mix_dist.kl_divergence(pi_dist)

        return stats
