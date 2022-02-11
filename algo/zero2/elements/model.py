import os
from typing import Tuple
import tensorflow as tf

from core.elements.model import Model
from core.tf_config import build
from utility.file import source_file

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class MAPPOModel(Model):
    def _pre_init(self):
        self.has_rnn = bool(self.config.get('rnn_type'))
        if self.config.encoder.nn_id.startswith('cnn'):
            self.config.encoder.time_distributed = 'rnn' in self.config

    def _build(self, env_stats, evaluation=False):
        n_units = len(env_stats.aid2uids[self.config.aid])
        basic_shape = (None, n_units)
        shapes = env_stats['obs_shape'][self.config.aid]
        dtypes = env_stats['obs_dtype'][self.config.aid]
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}

        if self.has_rnn:
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
            TensorSpecs.update(dict(
                state=self.state_type(*[((None, sz), dtype, name) 
                    for name, sz in self.state_size._asdict().items()]),
                mask=(basic_shape, tf.float32, 'mask'),
            ))
        TensorSpecs.update(dict(
            evaluation=evaluation,
            return_eval_stats=evaluation,
        ))
        self.action = build(self.action, TensorSpecs)

    @tf.function
    def action(
        self, 
        obs, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None,
        evaluation=False, 
        return_eval_stats=False
    ):
        if self.has_rnn:
            obs, mask = self._add_seqential_dimension(obs, mask)
        x, state = self.encode(obs, state=state, mask=mask)
        if self.has_rnn:
            x = tf.squeeze(x, 1)
        act_dist = self.policy(x, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)

        if evaluation:
            value = self.value(x)
            return action, {'value': value}, state
        else:
            logpi = act_dist.log_prob(action)
            value = self.value(x)
            terms = {'logpi': logpi, 'value': value}

            return action, terms, state    # keep the batch dimension for later use

    @tf.function
    def compute_value(
        self, 
        obs, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None
    ):
        x, state = self.encode(obs, state=state, mask=mask)
        value = self.value(x)
        return value, state

    def encode(
        self, 
        obs, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None
    ):
        if self.has_rnn:
            T, A = obs.shape[1:3]
            def merge_agent_dim(x):
                rank = len(x.shape)
                x = tf.transpose(x, [0, 2, 1, *range(3, rank)])
                x = tf.reshape(x, (-1, *x.shape[2:]))
                return x

            def restore_agent_dim(x):
                shape = x.shape
                x = tf.reshape(x, [-1, A, T, *shape[2:]])
                x = tf.transpose(x, [0, 2, 1, 3])
                return x

            obs, mask = tf.nest.map_structure(merge_agent_dim, (obs, mask))
            x = self.encoder(obs)
            if hasattr(self, 'rnn'):
                x, state = self.rnn(x, state, mask)
            x = restore_agent_dim(x)
        else:
            A = obs.shape[1]
            x = tf.reshape(obs, (-1, *obs.shape[2:]))
            x = self.encoder(x)
            x = tf.reshape(x, [-1, A, *x.shape[1:]])
            state = None

        return x, state

    def _add_seqential_dimension(
        self, 
        *args, 
        add_sequential_dim=True
    ):
        if add_sequential_dim:
            return tf.nest.map_structure(lambda x: tf.expand_dims(x, 1) 
                if isinstance(x, tf.Tensor) else x, args)
        else:
            return args


class MAPPGMixin:
    def _build(self, env_stats, evaluation=False):
        n_units = len(env_stats.aid2uids[self.config.aid])
        basic_shape = (None, n_units)
        shapes = env_stats['obs_shape'][self.config.aid]
        dtypes = env_stats['obs_dtype'][self.config.aid]
        TensorSpecs = {k: ((*basic_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}

        if self.has_rnn:
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
            TensorSpecs.update(dict(
                state=self.state_type(*[((None, sz), dtype, name) 
                    for name, sz in self.state_size._asdict().items()]),
                mask=(basic_shape, tf.float32, 'mask'),
            ))
        ActionTensorSpecs = dict(
            **TensorSpecs,
            evaluation=evaluation,
            return_eval_stats=evaluation,
        )
        self.action = build(self.action, ActionTensorSpecs)

        basic_shape = (None, self.config.sample_size, n_units)
        self.compute_logits_values = build(self.compute_logits_values, TensorSpecs)

    @tf.function
    def compute_logits_values(
        self,
        obs, 
        state: Tuple[tf.Tensor]=None,
        mask: tf.Tensor=None,
    ):
        x, state = self.encode(obs, state=state, mask=mask)
        act_dist = self.policy(x)
        value = self.value(x)
        return act_dist.logits, value


class MAPPGModel(MAPPGMixin, MAPPOModel):
    pass


def create_model(
        config, 
        env_stats, 
        name='mappo', 
        to_build=False,
        to_build_for_eval=False,
        **kwargs):
    aid = config['aid']
    config.policy.action_dim = env_stats.action_dim[aid]
    config.policy.is_action_discrete = env_stats.is_action_discrete[aid]

    Model = MAPPOModel if config.get('type', 'mappo') == 'mappo' else MAPPGModel
    return Model(
        config=config, 
        env_stats=env_stats, 
        name=name,
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        **kwargs
    )
