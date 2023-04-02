from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from core.log import do_logging, pwc
from core.elements.trainer import TrainerBase, create_trainer
from core import optimizer
from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.feature import one_hot
from tools.timer import Timer
from tools.utils import flatten_dict, prefix_name


def construct_fake_data(env_stats, aid):
    b = 8
    s = 400
    u = 2
    shapes = env_stats.obs_shape[aid]
    dtypes = env_stats.obs_dtype[aid]
    basic_shape = (b, s, u)
    data = {k: jnp.zeros((b, s+1, u, *v), dtypes[k]) 
        for k, v in shapes.items()}
    data = dict2AttrDict(data)
    data.setdefault('global_state', data.obs)
    action_dim = env_stats.action_dim[aid]
    data.action = jnp.zeros((*basic_shape, action_dim), jnp.float32)
    data.reset = jnp.zeros(basic_shape, jnp.float32)
    data.reward = jnp.zeros(basic_shape, jnp.float32)

    print_dict_info(data)
    
    return data


class Trainer(TrainerBase):
    def add_attributes(self):
        super().add_attributes()
        self._is_trust_worthy = not bool(self.config.trust_threshold)
    
    def is_trust_worthy(self):
        return self._is_trust_worthy

    def build_optimizers(self):
        theta = self.model.theta.copy()
        self.opts.theta, self.params.theta = optimizer.build_optimizer(
            params=theta, 
            **self.config.model_opt, 
            name='theta'
        )

    def train(self, data):
        data = self.process_data(data)
        theta = self.model.theta.copy()
        theta, self.params.theta, stats = \
            self.jit_train(
                theta, 
                opt_state=self.params.theta, 
                data=data, 
                return_stats=False
            )
        self.model.set_weights(theta)
        elite_indices = np.argsort(stats.mean_loss)
        self.model.rank_elites(elite_indices)

        data = flatten_dict(data, prefix='data')
        stats = prefix_name(stats, f'dynamics')
        stats.update(data)
        for v in theta.values():
            stats.update(flatten_dict(
                jax.tree_util.tree_map(np.linalg.norm, v), prefix='dynamics'))

        return stats

    def get_theta_params(self):
        weights = {
            'model': self.model.theta, 
            'opt': self.params.theta
        }
        return weights
    
    def set_theta_params(self, weights):
        self.model.set_weights(weights['model'])
        self.params.theta = weights['opt']

    def theta_train(
        self, 
        theta, 
        rng, 
        opt_state, 
        data, 
        return_stats=False
    ):
        do_logging('dynamics train is traced', backtrack=4)
        theta, opt_state, stats = optimizer.optimize(
            self.loss.loss, 
            theta, 
            opt_state, 
            kwargs={
                'rng': rng, 
                'data': data, 
            }, 
            opt=self.opts.theta, 
            name='train/dynamics'
        )

        if not return_stats:
            stats = AttrDict(
                model_mae=stats.model_mae, 
                obs_consistency=stats.obs_consistency, 
                reward_mae=stats.reward_mae, 
                discount_mae=stats.discount_mae, 
                trans_mae=stats.trans_mae, 
            )
        return theta, opt_state, stats

    def process_data(self, data):
        if self.env_stats.is_action_discrete[0]:
            data.action = one_hot(data.action, self.env_stats.action_dim[0])
        if self.model.config.model_norm_obs:
            data.obs_loc, data.obs_scale = \
                self.model.obs_rms.get_rms_stats(with_count=False)

        return data

    def update_rms(self, rms):
        if rms is not None:
            assert self.model.config.model_norm_obs, self.model.config.model_norm_obs
            self.model.obs_rms.update_from_moments(*rms)

    def _evaluate_model(self, stats):
        if self.config.trust_threshold is None:
            return
        if not self._is_trust_worthy:
            if 'model_mae' in stats:
                self._is_trust_worthy = np.mean(stats.model_mae) <= self.config.trust_threshold
            else:
                assert 'mean_loss' in stats, list(stats)
                self._is_trust_worthy = np.mean(stats.mean_loss) <= self.config.trust_threshold
        
    # def haiku_tabulate(self, data=None):
    #     rng = jax.random.PRNGKey(0)
    #     if data is None:
    #         data = construct_fake_data(self.env_stats, 0)
    #     print(hk.experimental.tabulate(self.theta_train)(
    #         self.model.theta, rng, self.params.theta, data
    #     ))


create_trainer = partial(create_trainer,
    name='model', trainer_cls=Trainer
)


def sample_stats(stats, max_record_size=10):
    # we only sample a small amount of data to reduce the cost
    stats = {k if '/' in k else f'train/{k}': 
        np.random.choice(v.reshape(-1), max_record_size) 
        if isinstance(v, (np.ndarray, jnp.DeviceArray)) else v 
        for k, v in stats.items()}
    return stats


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
