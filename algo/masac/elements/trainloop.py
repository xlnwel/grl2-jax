import jax

from tools.timer import timeit
from algo.ma_common.elements.trainloop import TrainingLoop as TrainingLoopBase
from algo.masac.elements.utils import compute_qs


class TrainingLoop(TrainingLoopBase):
    def _before_train(self, step):
        self.prev_params = self.model.theta.copy()

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
            self.model.theta.policies, rng, data)
        
        mu_params = [
            data[mk].reshape(data[mk].shape[0], -1) 
            for mk in mu_keys
        ]
        mu_dist = self.model.policy_dist(mu_params)
        kl = mu_dist.kl_divergence(pi_dist)

        stats.kl_mu_pi = kl

        q_data = data.slice((slice(None), slice(1)))
        mix_policies = [self.prev_params.policies[0]]
        mix_policies += self.model.params.policies[1:]
        mix_dist = self.model.joint_policy(
            mix_policies, rng, data
        )
        mix_actions = mix_dist.sample(seed=rng)
        q_mix = compute_qs(
            self.model.modules.Q, 
            self.model.params.Qs, 
            rng, 
            q_data.global_state, 
            mix_actions, 
            q_data.state_reset, 
            None, 
            return_minimum=True
        )
        stats.q_mix = q_mix
        mu_actions = mu_dist.sample(seed=rng)
        q_mu = compute_qs(
            self.model.modules.Q, 
            self.model.params.Qs, 
            rng, 
            q_data.global_state, 
            mu_actions, 
            q_data.state_reset, 
            None, 
            return_minimum=True
        )
        stats.q_mu = q_mu
        pi_actions = pi_dist.sample(seed=rng)
        q_pi = compute_qs(
            self.model.modules.Q, 
            self.model.params.Qs, 
            rng, 
            q_data.global_state, 
            pi_actions, 
            q_data.state_reset, 
            None, 
            return_minimum=True
        )
        stats.q_pi = q_pi

        stats.q_mix_mu_diff = q_mix - q_mu
        stats.q_pi_mu_diff = q_pi - q_mu

        return stats
