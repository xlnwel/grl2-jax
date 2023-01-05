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
        self.img_indices = np.arange(self.config.n_imaginary_envs)

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train)
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        
        _jit_img_train = jax.jit(self.img_train)
        def jit_img_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_img_train(*args, rng=rng, **kwargs)
        self.jit_img_train = jit_img_train
        
        self.haiku_tabulate()

    def imaginary_train(self, data: AttrDict):
        theta = self.model.imaginary_params.copy()
        is_imaginary = theta.pop('imaginary')
        assert is_imaginary == True, is_imaginary
        opt_state = self.imaginary_opt_state
        if self.config.popart:
            data.popart_mean = self.popart.mean
            data.popart_std = self.popart.std
        for _ in range(self.config.n_imaginary_epochs):
            np.random.shuffle(self.img_indices)
            indices = np.split(self.img_indices, self.config.n_imaginary_mbs)
            print(len(indices))
            from tools.display import print_dict_info
            print_dict_info(data)
            for idx in indices:
                with Timer('imaginary_train'):
                    d = data.slice(idx)
                    if self.config.popart:
                        d.popart_mean = self.popart.mean
                        d.popart_std = self.popart.std
                    theta, opt_state, _ = \
                        self.jit_img_train(
                            theta, 
                            opt_state=opt_state, 
                            data=data, 
                        )
        
        for k, v in theta.items():
            self.model.imaginary_params[k] = v
        self.imaginary_opt_state = opt_state

    def img_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
    ):
        do_logging('img train is traced', backtrack=4)
        if self.config.get('theta_opt'):
            theta, opt_state, stats = optimizer.optimize(
                self.loss.img_loss, 
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
                self.loss.img_value_loss, 
                theta.value, 
                opt_state.value, 
                kwargs={
                    'rng': rng, 
                    'data': data, 
                }, 
                opt=self.opts.value, 
                name='train/value'
            )
            theta.policy, opt_state.policy, stats = optimizer.optimize(
                self.loss.img_policy_loss, 
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
