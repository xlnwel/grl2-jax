from functools import partial
import numpy as np
import jax

from core.log import do_logging
from core.elements.trainer import create_trainer
from core.typing import AttrDict
from core import optimizer
from tools.timer import Timer
from algo.zero.elements.trainer import Trainer as TrainerBase


class Trainer(TrainerBase):
    def add_attributes(self):
        super().add_attributes()
        self.lka_indices = np.arange(self.config.n_lookahead_envs)

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

    def lookahead_train(self, data: AttrDict):
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
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, opt_state, _ = \
                        self.jit_lka_train(
                            theta, 
                            opt_state=opt_state, 
                            data=d, 
                        )
        
        for k, v in theta.items():
            self.model.lookahead_params[k] = v
        self.lookahead_opt_state = opt_state

    def lka_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
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
                    'stats': stats
                }, 
                opt=self.opts.policy, 
                name='train/policy'
            )

        return theta, opt_state, stats

create_trainer = partial(create_trainer,
    name='zero', trainer_cls=Trainer
)
