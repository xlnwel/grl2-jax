import jax

from tools.timer import timeit
from algo.lka_common.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def _before_train(self, step):
        self.prev_params = self.model.theta.copy()

    def train(self, step, **kwargs):
        self.lookahead_train(**kwargs)
        return super().train(step, **kwargs)

    @timeit
    def _after_train(self, stats):
        self.rng, rng = jax.random.split(self.rng, 2)
        BATCH_SIZE = 400
        mu_keys = self.model.policy_dist_stats('mu')
        data = self.buffer.sample_from_recency(
            batch_size=BATCH_SIZE, 
            sample_keys=['obs', *mu_keys], 
            n=BATCH_SIZE
        )
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
        mix_policies += self.model.params.policies[1:]
        mix_dist = self.model.joint_policy(
            mix_policies, rng, data
        )
        kl_mix_pi = mix_dist.kl_divergence(pi_dist)
        stats.kl_mix_pi = kl_mix_pi
        stats.kl_mu_mix_diff = kl_mu_pi - kl_mix_pi

        return stats
