import os
import collections
import tensorflow as tf

from core.elements.model import Model, ModelEnsemble
from core.tf_config import build
from utility.file import source_file
from utility.tf_utils import assert_rank

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class ModelImpl(Model):
    def encode(self, x, state, mask):
        # we expect x and mask to be of shape [B, T, A(, *)]
        assert_rank(x, 4)
        assert_rank(mask, 3)

        T, A, F = x.shape[1:]
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, T, F])
        mask = tf.transpose(mask, [0, 2, 1])
        mask = tf.reshape(mask, [-1, T])
        x = self.encoder(x)

        # merge the agent dimension into the batch dimension 
        state = tf.nest.map_structure(
            lambda x: tf.reshape(x, (-1, x.shape[-1])), state)
        if hasattr(self, 'rnn'):
            x, state = self.rnn(x, state, mask)
        else:
            state = None
        x = tf.reshape(x, [-1, A, T, x.shape[-1]])
        x = tf.transpose(x, [0, 2, 1, 3])

        return x, state


class MAPPOActorModel(ModelImpl):
    def action(self, obs, state, mask, action_mask=None,
            evaluation=False, return_eval_stats=False):
        x, state = self.encode(obs, state, mask)
        act_dist = self.policy(x, action_mask, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)

        if evaluation:
            # we do not compute the value state at evaluation 
            return action, {}, state
        else:
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi}
            return action, terms, state


class MAPPOValueModel(ModelImpl):
    def compute_value(self, global_state, state, mask):
        x, state = self.encode(global_state, state, mask)
        value = self.value(x)
        return value, state


class MAPPOModelEnsemble(ModelEnsemble):
    def _post_init(self):
        state = {
            'mlstm': 'actor_h actor_c value_h value_c',
            'mgru': 'actor_h value_h',
        }
        self.state_type = collections.namedtuple(
            'State', state[self._rnn_type.split('_')[-1]])
        self.compute_value = self.value.compute_value

    def _build(self, env_stats, evaluation=False):
        basic_shape = (None, len(env_stats.aid2pids[self.config.aid]))
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        TensorSpecs = dict(
            actor_inp=dict(
                obs=((*basic_shape, *env_stats['obs_shape']['obs']), 
                    env_stats['obs_dtype']['obs'], 'obs'),
                mask=(basic_shape, tf.float32, 'mask'),
            ),
            actor_state=self.actor_state_type(*[((None, sz), dtype, name) 
                for name, sz in self.actor_state_size._asdict().items()]),
            value_inp=dict(
                global_state=((*basic_shape, *env_stats['obs_shape']['global_state']), 
                    env_stats['obs_dtype']['global_state'], 'global_state'),
                mask=(basic_shape, tf.float32, 'mask'),
            ),
            value_state=self.value_state_type(*[((None, sz), dtype, name) 
                for name, sz in self.value_state_size._asdict().items()]),
            evaluation=evaluation,
            return_eval_stats=evaluation,
        )
        if env_stats.use_action_mask:
            TensorSpecs['actor_inp']['action_mask'] = (
                (*basic_shape, env_stats.action_dim), tf.bool, 'action_mask'
            )
        self.action = build(self.action, TensorSpecs)

    @tf.function
    def action(self, 
               actor_inp: dict, 
               actor_state: tuple, 
               value_inp: dict, 
               value_state: tuple,
               evaluation: bool=False, 
               return_eval_stats: bool=False):
        actor_inp, value_inp = self._add_seqential_dimension(actor_inp, value_inp)
        action, terms, actor_state = self.policy.action(
            **actor_inp, state=actor_state,
            evaluation=evaluation, return_eval_stats=return_eval_stats)
        value, value_state = self.value.compute_value(
            **value_inp, state=value_state)
        state = self.state_type(*actor_state, *value_state)
        if not evaluation:
            terms.update({
                'value': value,
            })
        action, terms = self._remove_seqential_dimension(action, terms)
        return action, terms, state

    def split_state(self, state):
        mid = len(state) // 2
        actor_state, value_state = state[:mid], state[mid:]
        return self.policy.state_type(*actor_state), \
            self.value.state_type(*value_state)

    @property
    def state_size(self):
        return self.state_type(*self.policy.rnn.state_size, *self.value.rnn.state_size)

    @property
    def actor_state_size(self):
        return self.policy.rnn.state_size

    @property
    def value_state_size(self):
        return self.value.rnn.state_size

    @property
    def state_keys(self):
        return self.state_type(*self.state_type._fields)

    @property
    def actor_state_keys(self):
        return self.policy.state_keys

    @property
    def value_state_keys(self):
        return self.value.state_keys

    @property
    def actor_state_type(self):
        return self.policy.state_type
    
    @property
    def value_state_type(self):
        return self.value.state_type

    def reset_states(self, state=None):
        actor_state, value_state = self.split_state(state)
        self.policy.rnn.reset_states(actor_state)
        self.value.rnn.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            inputs = inputs['obs']
        actor_state = self.policy.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        value_state = self.value.rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        return self.state_type(*actor_state, *value_state)

    def _add_seqential_dimension(self, *args):
        return tf.nest.map_structure(lambda x: tf.expand_dims(x, 1) 
            if isinstance(x, tf.Tensor) else x, args)

    def _remove_seqential_dimension(self, *args):
        return tf.nest.map_structure(lambda x: tf.squeeze(x, 1) 
            if isinstance(x, tf.Tensor) else x, args)


def create_model(
        config, 
        env_stats, 
        name='ppo', 
        to_build=False,
        to_build_for_eval=False,
        **kwargs):
    config.policy.policy.action_dim = env_stats.action_dim
    config.policy.policy.is_action_discrete = env_stats.action_dim

    return MAPPOModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        name=name,
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        policy=MAPPOActorModel,
        value=MAPPOValueModel,
        **kwargs
    )
