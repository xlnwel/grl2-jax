from functools import partial
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict
from tools.display import print_dict_info
from tools.timer import Timer
from tools.utils import flatten_dict, prefix_name
from algo.lka_common.elements.trainer import *
from algo.lka_common.elements.model import LOOKAHEAD


class Trainer(TrainerBase):
    def add_attributes(self):
        self.lookahead_theta = self.model.theta

    def build_optimizers(self):
        theta = self.model.theta.copy()
        self.params.theta = AttrDict()
        theta_policies, is_lookahead = pop_lookahead(theta.policies)
        assert all([lka == False for lka in is_lookahead]), is_lookahead
        self.opts.policy, self.params.theta.policy = optimizer.build_optimizer(
            params=theta_policies, 
            **self.config.policy_opt, 
            name='policy'
        )
        self.opts.Q, self.params.theta.Q = optimizer.build_optimizer(
            params=theta.Qs, 
            **self.config.Q_opt, 
            name='Q'
        )
        self.opts.temp, self.params.theta.temp = optimizer.build_optimizer(
            params=theta.temp, 
            **self.config.temp_opt, 
            name='temp'
        )
        self.lookahead_opt_state = self.params.theta

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train)
        def jit_train(*args, **kwargs):
            self.rng, rng = random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train

        _jit_lka_train = jax.jit(self.lka_train)
        def jit_lka_train(*args, **kwargs):
            self.rng, rng = random.split(self.rng)
            return _jit_lka_train(*args, rng=rng, **kwargs)
        self.jit_lka_train = jit_lka_train

        self.haiku_tabulate()

    def train(self, data: AttrDict):
        theta = self.model.theta.copy()
        theta.policies, is_lookahead = pop_lookahead(theta.policies)
        assert all([lka == False for lka in is_lookahead]), is_lookahead
        with Timer('theta_train'):
            theta, self.params.theta, stats = \
                self.jit_train(
                    theta, 
                    target_params=self.model.target_params, 
                    opt_state=self.params.theta, 
                    data=data,
                )

        for p in theta.policies:
            p[LOOKAHEAD] = False
        self.model.params = theta
        data = flatten_dict(data, prefix='data')
        stats = prefix_name(stats, 'theta')
        stats.update(data)
        self.model.update_target_params()

        return 1, stats

    def lookahead_train(self, data: AttrDict):
        theta = self.model.lookahead_params.copy()
        theta.policies, is_lookahead = pop_lookahead(theta.policies)
        assert all([lka == True for lka in is_lookahead]), is_lookahead
        opt_state = self.lookahead_opt_state
        theta, opt_state = self.jit_lka_train(
            theta, 
            target_params=self.model.theta, # we intend to use theta as the target network here.
            opt_state=opt_state, 
            data=data, 
        )
        
        for p in theta.policies:
            p[LOOKAHEAD] = True
        self.model.lookahead_params = theta
        self.lookahead_opt_state = opt_state

    def sync_lookahead_params(self):
        self.model.sync_lookahead_params()
        self.lookahead_opt_state = self.params.theta

    def theta_train(
        self, 
        theta, 
        rng, 
        target_params, 
        opt_state, 
        data, 
    ):
        do_logging('train is traced', backtrack=4)
        rngs = random.split(rng, 3)
        theta.Qs, opt_state.Q, stats = optimizer.optimize(
            self.loss.q_loss, 
            theta.Qs, 
            opt_state.Q, 
            kwargs={
                'rng': rngs[0], 
                'policy_params': theta.policies, 
                'target_qs_params': target_params.Qs, 
                'temp_params': theta.temp, 
                'data': data,
            }, 
            opt=self.opts.Q, 
            name='train/q'
        )
        theta.policies, opt_state.policy, stats = optimizer.optimize(
            self.loss.policy_loss, 
            theta.policies, 
            opt_state.policy, 
            kwargs={
                'rng': rngs[1], 
                'qs_params': theta.Qs, 
                'temp_params': theta.temp, 
                'data': data, 
                'stats': stats, 
            }, 
            opt=self.opts.policy, 
            name='train/policy'
        )
        if self.model.config.type != 'constant':
            theta.temp, opt_state.temp, stats = optimizer.optimize(
                self.loss.temp_loss, 
                theta.temp, 
                opt_state.temp, 
                kwargs={
                    'rng': rngs[2], 
                    'stats': stats
                }, 
                opt=self.opts.temp, 
                name='train/temp'
            )

        return theta, opt_state, stats

    def lka_train(
        self, 
        theta, 
        rng, 
        target_params, 
        opt_state, 
        data, 
    ):
        if self.config.lookahead_q:
            rng, q_rng = random.split(rng, 2)
            theta.Qs, opt_state.Q, stats = optimizer.optimize(
                self.loss.q_loss, 
                theta.Qs, 
                opt_state.Q, 
                kwargs={
                    'rng': q_rng, 
                    'policy_params': theta.policies, 
                    'target_qs_params': target_params.Qs, 
                    'temp_params': theta.temp, 
                    'data': data,
                }, 
                opt=self.opts.Q, 
                name='train/q'
            )
        else:
            stats = AttrDict()

        theta.policies, opt_state.policy, stats = optimizer.optimize(
            self.loss.policy_loss, 
            theta.policies, 
            opt_state.policy, 
            kwargs={
                'rng': rng, 
                'qs_params': theta.Qs, 
                'temp_params': theta.temp, 
                'data': data, 
                'stats': stats, 
            }, 
            opt=self.opts.policy, 
            name='train/policy'
        )

        return theta, opt_state

    # def haiku_tabulate(self, data=None):
    #     rng = jax.random.PRNGKey(0)
    #     if data is None:
    #         data = construct_fake_data(self.env_stats, 0)
    #     theta = self.model.theta.copy()
    #     is_lookahead = theta.pop(LOOKAHEAD)
    #     print(hk.experimental.tabulate(self.theta_train)(
    #         theta, rng, self.params.theta, data
    #     ))
    #     breakpoint()


create_trainer = partial(create_trainer,
    name='masac', trainer_cls=Trainer
)


if __name__ == '__main__':
    import haiku as hk
    from tools.yaml_op import load_config
    from env.func import create_env
    from .model import create_model
    from .loss import create_loss
    from core.log import pwc
    config = load_config('algo/ppo/configs/magw_a2c')
    config = load_config('distributed/sync/configs/smac')
    
    env = create_env(config.env)
    model = create_model(config.model, env.stats())
    loss = create_loss(config.loss, model)
    trainer = create_trainer(config.trainer, env.stats(), loss)
    data = construct_fake_data(env.stats(), 0)
    rng = jax.random.PRNGKey(0)
    pwc(hk.experimental.tabulate(trainer.jit_train)(
        model.theta, rng, trainer.params.theta, data), color='yellow')
    # data = construct_fake_data(env.stats(), 0, True)
    # pwc(hk.experimental.tabulate(trainer.raw_meta_train)(
    #     model.eta, model.theta, trainer.params, data), color='yellow')
