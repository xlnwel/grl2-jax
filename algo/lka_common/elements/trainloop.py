import jax

from core.log import do_logging
from core.elements.trainloop import TrainingLoop as TrainingLoopBase
from jax_tools.jax_div import js_from_distributions
from tools.timer import timeit


class TrainingLoop(TrainingLoopBase):
    def _before_train(self, step):
        self.prev_params = self.model.theta.copy()

    def train(self, step, **kwargs):
        train_step, stats = super().train(step, **kwargs)
        self.trainer.sync_lookahead_params()

        return train_step, stats

    def lookahead_train(self, **kwargs):
        if self.config.n_lka_epochs:
            for _ in range(self.config.n_lka_epochs):
                data = self.sample_data()
                if data is None:
                    do_logging('Bypassing lookahead train')
                    return

                self.trainer.lookahead_train(data, **kwargs)
        else:
            data = self.sample_data()
            if data is None:
                do_logging('Bypassing lookahead train')
                return

            self.trainer.lookahead_train(data, **kwargs)

    def compute_divs(self, rng, data, stats):
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

        js_mu_pi = js_from_distributions(
            **mu_dist.get_stats('p'), **pi_dist.get_stats('q'))
        js_lka_pi = js_from_distributions(
            **lka_dist.get_stats('p'), **pi_dist.get_stats('q'))
        js_mix_pi = js_from_distributions(
            **mix_dist.get_stats('p'), **pi_dist.get_stats('q'))
        stats.js_mu_pi = js_mu_pi
        stats.js_lka_pi = js_lka_pi
        stats.js_mix_pi = js_mix_pi
        stats.js_mu_lka_diff = js_mu_pi - js_lka_pi
        stats.js_mu_mix_diff = js_mu_pi - js_mix_pi

        return stats
