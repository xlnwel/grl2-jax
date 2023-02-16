from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from core.log import do_logging
from core.elements.trainer import create_trainer
from core.typing import AttrDict
from core import optimizer
from tools.timer import Timer
from algo.happo.elements.trainer import Trainer as TrainerBase


class Trainer(TrainerBase):
    def add_attributes(self):
        super().add_attributes()
        self.lka_indices = np.arange(self.config.n_simulated_envs)

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train)
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        
        _jit_lka_train = jax.jit(self.lka_train)
        def jit_lka_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_lka_train(*args, rng=rng, **kwargs)
        self.jit_lka_train = jit_lka_train
        
        self.haiku_tabulate()

    def lookahead_train(self, data: AttrDict, teammate_log_ratio=None):
        if teammate_log_ratio is None:
            teammate_log_ratio = jnp.zeros_like(data.mu_logprob)

        theta = self.model.lookahead_params.copy()
        is_lookahead = theta.pop('lookahead')
        assert is_lookahead == True, is_lookahead
        opt_state = self.lookahead_opt_state
        for _ in range(self.config.n_lookahead_epochs):
            np.random.shuffle(self.lka_indices)
            indices = np.split(self.lka_indices, self.config.n_lookahead_mbs)
            for idx in indices:
                with Timer('lookahead_train'):
                    d = data.slice(idx)
                    t_log_ratio = teammate_log_ratio[idx]
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, self.params.theta, _ = \
                        self.jit_train(
                            theta, 
                            opt_state=self.params.theta, 
                            data=d,
                            teammate_log_ratio=t_log_ratio,
                        )
        
        self.model.set_weights(theta)

        self.rng, rng = jax.random.split(self.rng) 
        pi_logprob = self.model.jit_action_logprob(
            self.model.lookahead_params, rng, data)
        agent_log_ratio = pi_logprob - data.mu_logprob
        return teammate_log_ratio + agent_log_ratio

    def lka_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
        teammate_log_ratio,
    ):
        do_logging('lka train is traced', backtrack=4)
        if self.config.get('theta_opt'):
            theta, opt_state, stats = optimizer.optimize(
                self.loss.lka_loss, 
                theta, 
                opt_state, 
                kwargs={
                    'rng': rng, 
                    'data': data,
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.theta, 
                name='train/theta'
            )
        else:
            theta.value, opt_state.value, stats = optimizer.optimize(
                self.loss.lka_value_loss, 
                theta.value, 
                opt_state.value, 
                kwargs={
                    'rng': rng, 
                    'policy_theta': theta.policy, 
                    'data': data,
                }, 
                opt=self.opts.value, 
                name='train/value'
            )
            theta.policy, opt_state.policy, stats = optimizer.optimize(
                self.loss.lka_policy_loss, 
                theta.policy, 
                opt_state.policy, 
                kwargs={
                    'rng': rng, 
                    'data': data, 
                    'stats': stats,
                    'teammate_log_ratio': teammate_log_ratio,
                }, 
                opt=self.opts.policy, 
                name='train/policy'
            )

        return theta, opt_state, stats

create_trainer = partial(create_trainer,
    name='happo_mb', trainer_cls=Trainer
)
