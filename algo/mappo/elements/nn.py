import collections
import tensorflow as tf

from utility.tf_utils import assert_rank
from core.module import Ensemble
from algo.ppo.elements.nn import *


class PPO(Ensemble):
    def __init__(self, config, env, model_fn=create_components, **kwargs):
        state = {
            'lstm': 'actor_h actor_c value_h value_c',
            'mlstm': 'actor_h actor_c value_h value_c',
            'gru': 'actor_h value_h',
            'mgru': 'actor_h value_h',
        }
        self.State = collections.namedtuple(
            'State', state[config['actor_rnn']['nn_id'].split('_')[1]])
        
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, global_state, 
            state, mask, action_mask=None,
            evaluation=False, prev_action=None, 
            prev_reward=None, **kwargs):
        assert obs.shape.ndims % 2 == 0, obs.shape

        actor_state, value_state = self.split_state(state)
        x_actor, actor_state = self.encode(
            obs, actor_state, mask, 'actor', 
            prev_action, prev_reward)
        act_dist = self.actor(x_actor, action_mask, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)

        if evaluation:
            # we do not compute the value state at evaluation 
            return action, self.State(*actor_state, *value_state)
        else:
            x_value, value_state = self.encode(
                global_state, value_state, mask, 'value', 
                prev_action, prev_reward)
            value = self.value(x_value)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            out = (action, terms)
            return out, self.State(*actor_state, *value_state)

    @tf.function(experimental_relax_shapes=True)
    def compute_value(self, global_state, state, mask, 
            prev_action=None, prev_reward=None, return_state=False):
        x, state = self.encode(
            global_state, state, mask, 'value', prev_action, prev_reward)
        value = self.value(x)
        if return_state:
            return value, state
        else:
            return value

    def encode(self, x, state, mask, stream, prev_action=None, prev_reward=None):
        assert stream in ('actor', 'value'), stream
        if stream == 'actor':
            encoder = self.actor_encoder
            rnn = self.actor_rnn
        else:
            encoder = self.value_encoder
            rnn = self.value_rnn
        if x.shape.ndims % 2 == 0:
            x = tf.expand_dims(x, 1)
        if mask.shape.ndims < 2:
            mask = tf.reshape(mask, (-1, 1))
        assert_rank(mask, 2)

        x = encoder(x)
        additional_rnn_input = self._process_additional_input(
            x, prev_action, prev_reward)
        x, state = rnn(x, state, mask, 
            additional_input=additional_rnn_input)
        if x.shape[1] == 1:
            x = tf.squeeze(x, 1)
        return x, state

    def _process_additional_input(self, x, prev_action, prev_reward):
        results = []
        if prev_action is not None:
            if self.actor.is_action_discrete:
                if prev_action.shape.ndims < 2:
                    prev_action = tf.reshape(prev_action, (-1, 1))
                prev_action = tf.one_hot(
                    prev_action, self.actor.action_dim, dtype=x.dtype)
            else:
                if prev_action.shape.ndims < 3:
                    prev_action = tf.reshape(
                        prev_action, (-1, 1, self.actor.action_dim))
            assert_rank(prev_action, 3)
            results.append(prev_action)
        if prev_reward is not None:
            if prev_reward.shape.ndims < 2:
                prev_reward = tf.reshape(prev_reward, (-1, 1, 1))
            elif prev_reward.shape.ndims == 2:
                prev_reward = tf.expand_dims(prev_reward, -1)
            assert_rank(prev_reward, 3)
            results.append(prev_reward)
        assert_rank(results, 3)
        return results

    def split_state(self, state):
        mid = len(state) // 2
        actor_state, value_state = state[:mid], state[mid:]
        return actor_state, value_state

    def reset_states(self, state=None):
        actor_state, value_state = self.split_state(state)
        self.actor_rnn.reset_states(actor_state)
        self.value_rnn.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        actor_state = self.actor_rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        value_state = self.value_rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype)
        return self.State(*actor_state, *value_state)

    @property
    def state_size(self):
        return self.State(*self.actor_rnn.state_size, *self.value_rnn.state_size)

    @property
    def actor_state_size(self):
        return self.actor_rnn.state_size

    @property
    def value_state_size(self):
        return self.value_rnn.state_size

    @property
    def state_keys(self):
        return self.State(*self.State._fields)


def create_model(config, env, **kwargs):
    return PPO(config, env, create_components, **kwargs)
