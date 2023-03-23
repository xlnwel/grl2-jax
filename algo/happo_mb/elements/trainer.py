from functools import partial
import numpy as np
import jax

from core.log import do_logging
from core.elements.trainer import create_trainer
from core.typing import AttrDict
from core import optimizer
from algo.lka_common.elements.model import LOOKAHEAD, pop_lookahead
from algo.happo.elements.trainer import Trainer as TrainerBase


class Trainer(TrainerBase):
    def add_attributes(self):
        super().add_attributes()
        self.lka_indices = np.arange(self.config.n_simulated_envs)

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train, 
            static_argnames=['aid', 'compute_teammate_log_ratio'])
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        
        _jit_lka_train = jax.jit(self.lka_train, 
            static_argnames=['aid', 'compute_teammate_log_ratio'])
        def jit_lka_train(*args, **kwargs):
            self.lka_rng, rng = jax.random.split(self.lka_rng)
            return _jit_lka_train(*args, rng=rng, **kwargs)
        self.jit_lka_train = jit_lka_train
        
        self.haiku_tabulate()

    def lookahead_train(self, data: AttrDict):
        theta = self.model.lookahead_params.copy()
        theta.policies, is_lookahead = pop_lookahead(theta.policies)
        assert all([lka == True for lka in is_lookahead]), is_lookahead
        opt_state = self.lookahead_opt_state

        if self.config.update_scheme == 'step':
            theta, opt_state, self.perm_lka_rng = self.stepwise_sequential_opt(
                theta, opt_state, data, self.config.n_lka_epochs, 
                self.config.n_lka_mbs, self.lka_indices, 
                self.jit_lka_train, self.perm_lka_rng, 
                return_stats=False
            )
        else:
            theta, opt_state, self.perm_lka_rng = self.sequential_opt(
                theta, opt_state, data, self.config.n_lka_epochs, 
                self.config.n_lka_mbs, self.lka_indices, 
                self.jit_lka_train, self.perm_lka_rng, 
                return_stats=False
            )

        for p in theta.policies:
            p[LOOKAHEAD] = True
        self.model.set_lka_params(theta)
        self.lookahead_opt_state = opt_state

    def lka_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
        teammate_log_ratio,
        aid, 
        compute_teammate_log_ratio=True
    ):
        do_logging('lka train is traced', backtrack=4)
        rngs = jax.random.split(rng, 3)
        if self.config.get('theta_opt'):
            theta, opt_state, stats = optimizer.optimize(
                self.loss.lka_loss, 
                theta, 
                opt_state, 
                kwargs={
                    'rng': rngs[0], 
                    'data': data,
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.theta[aid], 
                name='train/theta'
            )
        else:
            theta.value, opt_state.value, stats = optimizer.optimize(
                self.loss.lka_value_loss, 
                theta.value, 
                opt_state.value, 
                kwargs={
                    'rng': rngs[0], 
                    'policy_theta': theta.policy, 
                    'data': data,
                }, 
                opt=self.opts.vs[aid], 
                name='train/value'
            )
            theta.policy, opt_state.policy, stats = optimizer.optimize(
                self.loss.lka_policy_loss, 
                theta.policy, 
                opt_state.policy, 
                kwargs={
                    'rng': rngs[1], 
                    'data': data, 
                    'stats': stats,
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.policies[aid], 
                name='train/policy'
            )

        if compute_teammate_log_ratio:
            stats.teammate_log_ratio = self.compute_teammate_log_ratio(
                theta.policy, rngs[2], teammate_log_ratio, data)

        return theta, opt_state, stats


create_trainer = partial(create_trainer,
    name='happo_mb', trainer_cls=Trainer
)
