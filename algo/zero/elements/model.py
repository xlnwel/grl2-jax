import os
from typing import Tuple
import tensorflow as tf

from core.elements.model import Model as ModelBase, ModelEnsemble as ModelEnsembleBase
from core.mixin.model import NetworkSyncOps
from core.tf_config import build
from utility.file import source_file
from utility.typing import AttrDict
from .utils import compute_inner_steps

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class Model(ModelBase):
    def _pre_init(self):
        self.config = compute_inner_steps(self.config)
        self.has_rnn = bool(self.config.get('rnn_type'))
        if self.config.encoder.nn_id is not None \
            and self.config.encoder.nn_id.startswith('cnn'):
            self.config.encoder.time_distributed = 'rnn' in self.config

    def _build(
        self, 
        env_stats: AttrDict, 
        evaluation: bool=False
    ):
        aid = self.config.get('aid', 0)
        basic_shape = (None, len(env_stats.aid2uids[aid]))
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        shapes = env_stats['obs_shape'][aid]
        dtypes = env_stats['obs_dtype'][aid]
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}

        if self.has_rnn:
            TensorSpecs.update(dict(
                state=self.state_type(*[((None, sz), dtype, name) 
                    for name, sz in self.state_size._asdict().items()]),
                mask=(basic_shape, tf.float32, 'mask'),
                evaluation=evaluation,
                return_eval_stats=evaluation,
            ))
        if env_stats.use_action_mask:
            TensorSpecs['action_mask'] = (
                (*basic_shape, env_stats.action_dim[aid]), tf.bool, 'action_mask'
            )
        if env_stats.use_life_mask:
            TensorSpecs['life_mask'] = (
                basic_shape, tf.float32, 'life_mask'
            )
        self.action = build(self.action, TensorSpecs)

    def _post_init(self):
        self.has_rnn = bool(self.config.get('rnn_type'))

    @tf.function
    def action(
        self, 
        obs, 
        idx=None, 
        global_state=None, 
        hidden_state=None, 
        action_mask=None, 
        life_mask=None, 
        prev_reward=None,
        prev_action=None,
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None,
        evaluation=False, 
        return_eval_stats=False
    ):
        x, state = self.encode(obs, state=state, mask=mask)
        act_dist = self.policy(x, idx=idx, action_mask=action_mask, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)

        if self.policy.is_action_discrete:
            pi = tf.nn.softmax(act_dist.logits)
            terms = {
                'mu': pi
            }
        else:
            mean = act_dist.mean()
            std = tf.exp(self.policy.logstd)
            terms = {
                'mu_mean': mean,
                'mu_std': std * tf.ones_like(mean), 
            }

        if global_state is None:
            global_state = x
        if evaluation:
            value = self.value(global_state, idx)
            return action, {'value': value}, state
        else:
            logprob = act_dist.log_prob(action)
            tf.debugging.assert_all_finite(logprob, 'Bad logprob')
            value = self.value(global_state, idx)
            terms.update({'mu_logprob': logprob, 'value': value})

            return action, terms, state    # keep the batch dimension for later use

    def forward(
        self, 
        obs, 
        idx=None, 
        global_state=None, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None,
    ):
        x, state = self.encode(obs, state=state, mask=mask)
        act_dist = self.policy(x, idx)
        if global_state is None:
            global_state = x
        value = self.value(global_state, idx)
        return x, act_dist, value

    @tf.function
    def compute_value(
        self, 
        obs, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None
    ):
        shape = obs.shape
        x = tf.reshape(obs, [-1, *shape[2:]])
        x, state = self.encode(x, state, mask)
        value = self.value(x)
        value = tf.reshape(value, (-1, shape[1]))
        return value, state

    def encode(
        self, 
        x, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None
    ):
        tf.debugging.assert_all_finite(x, 'Bad input')
        x = self.encoder(x)
        tf.debugging.assert_all_finite(x, 'Bad encoder output')
        use_meta = self.config.inner_steps is not None
        if use_meta and hasattr(self, 'embed'):
            gamma = self.meta('gamma', inner=use_meta)
            lam = self.meta('lam', inner=use_meta)
            tf.debugging.assert_all_finite(gamma, 'Bad gamma')
            tf.debugging.assert_all_finite(lam, 'Bad lam')
            x = self.embed(x, gamma, lam)
            tf.debugging.assert_all_finite(x, 'Bad embedding')
        if hasattr(self, 'rnn'):
            x, state = self.rnn(x, state, mask)
            return x, state
        else:
            return x, None


class ModelEnsemble(ModelEnsembleBase):
    def _pre_init(self):
        self.config = compute_inner_steps(self.config)

    def _post_init(self):
        self.sync_ops = NetworkSyncOps()
        
        model = self.meta if self.config.inner_steps == 1 and self.config.extra_meta_step == 0 else self.rl
        self.state_size = model.state_size
        self.state_keys = model.state_keys
        self.state_type = model.state_type
        self.get_states = model.get_states
        self.reset_states = model.reset_states
        self.action = model.action

    def sync_nets(self, forward=True):
        if self.config.inner_steps is not None:
            self.sync_meta_nets()
            if forward:
                self.sync_meta_rl_nets()
            else:
                self.sync_rl_meta_nets()

    @tf.function
    def sync_meta_nets(self):
        keys = sorted([k for k in self.meta.keys() if k.startswith('meta')])
        source = [self.meta[k] for k in keys]
        target = [self.rl[k] for k in keys]
        self.sync_ops.sync_nets(source, target)

    @tf.function
    def sync_rl_meta_nets(self):
        keys = sorted([k for k in self.meta.keys() if not k.startswith('meta')])
        source = [self.rl[k] for k in keys]
        target = [self.meta[k] for k in keys]
        self.sync_ops.sync_nets(source, target)

    @tf.function
    def sync_meta_rl_nets(self):
        keys = sorted([k for k in self.meta.keys() if not k.startswith('meta')])
        source = [self.meta[k] for k in keys]
        target = [self.rl[k] for k in keys]
        self.sync_ops.sync_nets(source, target)


def create_model(
    config, 
    env_stats, 
    name='zero', 
    to_build=False, 
    to_build_for_eval=False, 
    **kwargs
):
    if 'aid' in config:
        aid = config['aid']
        config.policy.action_dim = env_stats.action_dim[aid]
        config.policy.is_action_discrete = env_stats.is_action_discrete[aid]
    else:
        config.policy.action_dim = env_stats.action_dim
        config.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.action_low = env_stats.get('action_low')
        config.policy.action_high = env_stats.get('action_high')

    if config['rnn_type'] is None:
        config.pop('rnn', None)
    else:
        config['rnn']['nn_id'] = config['actor_rnn_type']

    rl = Model(
        config=config, 
        env_stats=env_stats, 
        name='rl',
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        **kwargs
    )
    meta = Model(
        config=config, 
        env_stats=env_stats, 
        name='meta',
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        **kwargs
    )
    return ModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        components=dict(
            rl=rl, 
            meta=meta, 
        ), 
        name=name, 
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        **kwargs
    )
