import jax

from tools.timer import timeit
from algo.lka_common.elements.trainloop import TrainingLoop as TrainingLoopBase
from algo.masac.elements.utils import compute_qs


class TrainingLoop(TrainingLoopBase):
    @timeit
    def _after_train(self, stats):
        self.rng, rng = jax.random.split(self.rng, 2)
        BATCH_SIZE = 400
        mu_keys = self.model.policy_dist_stats('mu')
        data = self.buffer.sample_from_recency(
            batch_size=BATCH_SIZE, 
            sample_keys=['obs', 'global_state', *mu_keys], 
            n=BATCH_SIZE
        )
        if data is None:
            return stats
        
        pi_dist = self.model.joint_policy(
            self.model.theta.policies, rng, data
        )

        mu_params = [
            data[mk].reshape(*data[mk].shape[:-2], 1, -1) 
            for mk in mu_keys
        ]
        mu_dist = self.model.policy_dist(mu_params)
        kl_mu_pi = mu_dist.kl_divergence(pi_dist)

        lka_dist = self.model.joint_policy(
            self.model.lookahead_params.policies, rng, data
        )
        kl_lka_pi = lka_dist.kl_divergence(pi_dist)

        stats.kl_mu_pi = kl_mu_pi
        stats.kl_lka_pi = kl_lka_pi
        stats.kl_mu_lka_diff = kl_mu_pi - kl_lka_pi

        mix_policies = [self.prev_params.policies[0]]
        mix_policies += self.model.lookahead_params.policies[1:]
        mix_dist = self.model.joint_policy(
            mix_policies, rng, data
        )
        mix_actions = mix_dist.sample(seed=rng)
        q_data = data.slice((slice(None), slice(1)))
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
        lka_actions = lka_dist.sample(seed=rng)
        q_lka = compute_qs(
            self.model.modules.Q, 
            self.model.params.Qs, 
            rng, 
            q_data.global_state, 
            lka_actions, 
            q_data.state_reset, 
            None, 
            return_minimum=True
        )
        stats.q_lka = q_lka
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
        stats.q_lka_mu_diff = q_lka - q_mu
        stats.q_pi_mu_diff = q_pi - q_mu

        return stats
