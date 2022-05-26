import os
import collections
import tensorflow as tf

from core.elements.model import Model, ModelEnsemble
from core.tf_config import build
from utility.file import source_file
from utility.tf_utils import assert_rank

source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class ModelImpl(Model):
    def encode(self, x, prev_reward, prev_action, state, mask):
        has_rnn = hasattr(self, 'rnn')
        if has_rnn:
            if self.env_stats.is_multi_agent:
                # we expect x and mask to be of shape [B, T, U(, *)]
                assert_rank(x, 4)
                assert_rank(prev_reward, 3)
                assert_rank(prev_action, 4)
                assert_rank(mask, 3)

                T, U, F = x.shape[1:]
                assert prev_reward.shape[1:] == (T, U), (prev_reward.shape, (T, U))
                assert mask.shape[1:] == (T, U), (mask.shape, (T, U))
                x = tf.transpose(x, [0, 2, 1, 3])
                x = tf.reshape(x, [-1, T, F])
                mask = tf.transpose(mask, [0, 2, 1])
                mask = tf.reshape(mask, [-1, T])
                x = self.encoder(x)
                if self.config.use_prev_reward:
                    prev_reward = tf.transpose(prev_reward, [0, 2, 1])
                    prev_reward = tf.reshape(prev_reward, [-1, T])
                    prev_reward = prev_reward * mask
                    x = tf.concat([x, prev_reward[..., None]], -1)
                if self.config.use_prev_action:
                    A = prev_action.shape[-1]
                    prev_action = tf.transpose(prev_action, [0, 2, 1, 3])
                    prev_action = tf.reshape(prev_action, [-1, T, A])
                    prev_action = prev_action * mask[..., None]
                    x = tf.concat([x, prev_action], -1)
                x, state = self.rnn(x, state, mask)
                x = tf.reshape(x, [-1, U, T, x.shape[-1]])
                x = tf.transpose(x, [0, 2, 1, 3])
            else:
                # we expect x and mask to be of shape [B, T(, *)]
                assert_rank(x, 3)
                assert_rank(prev_reward, 2)
                assert_rank(prev_action, 3)
                assert_rank(mask, 2)

                T, F = x.shape[1:]
                assert prev_reward.shape[1:] == (T,), (prev_reward.shape, (T,))
                assert mask.shape[1:] == (T,), (mask.shape, (T,))
                x = self.encoder(x)
                if self.config.use_prev_reward:
                    prev_reward = prev_reward * mask
                    x = tf.concat([x, prev_reward[..., None]], -1)
                if self.config.use_prev_action:
                    prev_action = prev_action * mask[..., None]
                    x = tf.concat([x, prev_action], -1)
                x, state = self.rnn(x, state, mask)
                x = tf.reshape(x, [-1, T, x.shape[-1]])
        else:
            # if self.env_stats.is_multi_agent:
            #     assert_rank(x, 3)
            # else:
            #     assert_rank(x, 2)

            x = self.encoder(x)

            state = None

        return x, state


class PPOActorModel(ModelImpl):
    def action(
        self, 
        obs, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        mask=None, 
        action_mask=None,
        evaluation=False, 
        return_eval_stats=False
    ):
        x, state = self.encode(
            x=obs, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )
        act_dist = self.policy(x, action_mask, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)
        if self.policy.is_action_discrete:
            pi = tf.nn.softmax(act_dist.logits)
            terms = {
                'pi': pi
            }
        else:
            mean = act_dist.mean()
            std = tf.exp(self.policy.logstd)
            terms = {
                'pi_mean': mean,
                'pi_std': std * tf.ones_like(mean), 
            }
        if evaluation:
            # we do not compute the value state at evaluation 
            return action, {}, state
        else:
            logprob = act_dist.log_prob(action)
            terms['logprob'] = logprob
            return action, terms, state


class PPOValueModel(ModelImpl):
    @tf.function
    def compute_value(
        self, 
        global_state, 
        prev_reward=None, 
        prev_action=None, 
        life_mask=None, 
        state=None, 
        mask=None
    ):
        x, state = self.encode(
            x=global_state, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=state, 
            mask=mask
        )
        value = self.value(x)
        return value, state


class PPOModelEnsemble(ModelEnsemble):
    def _build(self, env_stats, evaluation=False):
        aid = self.config.aid
        basic_shape = (None, len(env_stats.aid2uids[aid]))
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        actor_inp=dict(
            obs=((*basic_shape, *env_stats.obs_shape[aid]['obs']), 
                env_stats.obs_dtype[aid]['obs'], 'obs'),
            prev_reward=(basic_shape, tf.float32, 'prev_reward'), 
            prev_action=((*basic_shape, env_stats.action_dim[aid]), tf.float32, 'prev_action'), 
        )
        value_inp=dict(
            global_state=(
                (*basic_shape, *env_stats.obs_shape[aid]['global_state']), 
                env_stats.obs_dtype[aid]['global_state'], 'global_state'), 
            prev_reward=(basic_shape, tf.float32, 'prev_reward'), 
            prev_action=((*basic_shape, env_stats.action_dim[aid]), tf.float32, 'prev_action'), 
        )
        TensorSpecs = dict(
            actor_inp=actor_inp,
            value_inp=value_inp,
            evaluation=evaluation,
            return_eval_stats=evaluation,
        )
        if self.has_rnn:
            for inp in [actor_inp, value_inp]:
                inp['mask'] = (basic_shape, tf.float32, 'mask')
            if self.actor_has_rnn:
                TensorSpecs['actor_state'] = self.actor_state_type(
                    *[((None, sz), dtype, name) 
                    for name, sz in self.actor_state_size._asdict().items()])
            if self.value_has_rnn:
                TensorSpecs['value_state'] = self.value_state_type(
                    *[((None, sz), dtype, name) 
                    for name, sz in self.value_state_size._asdict().items()])
        if env_stats.use_action_mask:
            TensorSpecs['actor_inp']['action_mask'] = (
                (*basic_shape, env_stats.action_dim[aid]), tf.bool, 'action_mask'
            )
        if env_stats.use_life_mask:
            TensorSpecs['value_inp']['life_mask'] = (
                basic_shape, tf.float32, 'life_mask'
            )
        self.action = build(self.action, TensorSpecs)

    def _post_init(self):
        self.actor_has_rnn = bool(self.config.get('actor_rnn_type'))
        self.value_has_rnn = bool(self.config.get('value_rnn_type'))
        self.has_rnn = self.actor_has_rnn or self.value_has_rnn
        if self.has_rnn:
            self.state_type = collections.namedtuple('State', 'actor value')
        else:
            self.state_type = None

        self.compute_value = self.value.compute_value

    @tf.function
    def action(
        self, 
        actor_inp: dict, 
        value_inp: dict, 
        actor_state: tuple=None, 
        value_state: tuple=None, 
        evaluation: bool=False, 
        return_eval_stats: bool=False
    ):
        if self.actor_has_rnn:
            actor_inp, = self._add_seqential_dimension(actor_inp)
        action, terms, actor_state = self.policy.action(
            **actor_inp, state=actor_state,
            evaluation=evaluation, return_eval_stats=return_eval_stats)
        state = self.state_type(actor_state, value_state) \
            if self.has_rnn else None
        if self.actor_has_rnn:
            action, terms = self._remove_seqential_dimension(action, terms)
        if self.config.get('compute_value_at_execution', True) or self.has_rnn:
            if self.value_has_rnn:
                value_inp, = self._add_seqential_dimension(value_inp)
            value, value_state = self.value.compute_value(
                **value_inp, state=value_state)
            state = self.state_type(actor_state, value_state) \
                if self.has_rnn else None
            if self.value_has_rnn:
                value, = self._remove_seqential_dimension(value)
            if not evaluation:
                terms.update({
                    'value': value,
                })
        return action, terms, state

    def split_state(self, state):
        if self.has_rnn:
            actor_state, value_state = state
            if self.actor_has_rnn:
                assert actor_state is not None, actor_state
                actor_state = self.actor_state_type(*actor_state)
            if self.value_has_rnn:
                assert value_state is not None, value_state
                value_state = self.value_state_type(*value_state)
            return actor_state, value_state

    @property
    def state_size(self):
        if self.state_type is None:
            return None
        return self.state_type(self.actor_state_size, self.value_state_size)

    @property
    def actor_state_size(self):
        return self.policy.state_size if self.actor_has_rnn else None

    @property
    def value_state_size(self):
        return self.value.state_size if self.value_has_rnn else None

    @property
    def state_keys(self):
        if self.state_type is None:
            return ()
        return self.state_type(*self.state_type._fields)

    @property
    def actor_state_keys(self):
        return self.policy.state_keys if self.actor_has_rnn else ()

    @property
    def value_state_keys(self):
        return self.value.state_keys if self.value_has_rnn else ()

    @property
    def actor_state_type(self):
        return self.policy.state_type if self.actor_has_rnn else None
    
    @property
    def value_state_type(self):
        return self.value.state_type if self.value_has_rnn else None

    def get_states(self):
        actor_state = self.policy.get_states()
        value_state = self.value.get_states()
        if actor_state is not None or value_state is not None:
            return self.state_type(actor_state, value_state)
        return None

    def reset_states(self, state=None):
        if state is not None:
            actor_state, value_state = self.split_state(state)
            self.policy.reset_states(actor_state)
            self.value.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            inputs = inputs['obs']
        if self.state_type:
            actor_state = self.policy.get_initial_state(
                inputs, batch_size=batch_size, dtype=dtype) \
                    if self.actor_has_rnn else None
            value_state = self.value.get_initial_state(
                inputs, batch_size=batch_size, dtype=dtype) \
                    if self.value_has_rnn else None
            return self.state_type(actor_state, value_state)
        return None

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
    **kwargs
):
    if 'aid' in config:
        aid = config['aid']
        config.policy.policy.action_dim = env_stats.action_dim[aid]
        config.policy.policy.is_action_discrete = env_stats.is_action_discrete[aid]
        config.policy.policy.action_low = env_stats.get('action_low')
        config.policy.policy.action_high = env_stats.get('action_high')
    else:
        config.policy.policy.action_dim = env_stats.action_dim
        config.policy.policy.is_action_discrete = env_stats.is_action_discrete
        config.policy.policy.action_low = env_stats.get('action_low')
        config.policy.policy.action_high = env_stats.get('action_high')

    if config['actor_rnn_type'] is None:
        config['policy'].pop('rnn', None)
    else:
        config['policy']['rnn']['nn_id'] = config['actor_rnn_type']
    if config['value_rnn_type'] is None:
        config['value'].pop('rnn', None)
    else:
        config['value']['rnn']['nn_id'] = config['value_rnn_type']

    return PPOModelEnsemble(
        config=config, 
        env_stats=env_stats, 
        name=name,
        to_build=to_build, 
        to_build_for_eval=to_build_for_eval,
        policy=PPOActorModel,
        value=PPOValueModel,
        **kwargs
    )
