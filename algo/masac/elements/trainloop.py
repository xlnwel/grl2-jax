import jax

from tools.timer import timeit
from algo.ma_common.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    @timeit
    def ma_policy_stats(self, stats):
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
        
        pi_params = self.model.joint_policy(
            self.model.theta.policies, rng, data)
        pi_dist = self.model.policy_dist(pi_params)

        mu_params = [
            data[mk].reshape(data[mk].shape[0], -1) 
            for mk in mu_keys
        ]
        mu_dist = self.model.policy_dist(mu_params)
        kl = mu_dist.kl_divergence(pi_dist)

        stats.kl_mu_pi = kl

        return stats
