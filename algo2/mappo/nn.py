import collections
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.tf_utils import assert_rank
from core.module import Module, Ensemble
from nn.func import Encoder, GRU, mlp
from algo.ppo.nn import Actor, Value

State = collections.namedtuple('State', 'actor_h value_h')


class Actor(Module):
    def __init__(self, config, action_dim, is_action_discrete, name='actor'):
        super().__init__(name=name)
        config = config.copy()

        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        self.eval_act_temp = config.pop('eval_act_temp', 1)
        assert self.eval_act_temp >= 0, self.eval_act_temp

        self._init_std = config.pop('init_std', 1)
        if not self.is_action_discrete:
            self.logstd = tf.Variable(
                initial_value=np.log(self._init_std)*np.ones(action_dim), 
                dtype='float32', 
                trainable=True, 
                name=f'actor/logstd')
        config.setdefault('out_gain', .01)
        self._layers = mlp(**config, 
                        out_size=action_dim, 
                        out_dtype='float32',
                        name=name)

    def call(self, x, action_mask, evaluation=False):
        actor_out = self._layers(x)

        logits = actor_out / self.eval_act_temp \
            if evaluation and self.eval_act_temp else actor_out
        assert logits.shape[1:] == action_mask.shape[1:], (logits.shape, action_mask.shape)
        logits = tf.where(action_mask, logits, -1e10)
        act_dist = tfd.Categorical(logits)

        return act_dist

    def action(self, dist, action_mask, evaluation):
        if evaluation:
            action = dist.mode()
        else:
            action = dist.sample()
            # ensures all actions are valid. This is time-consuming, and we opt to allow invalid actions
            # def cond(a, x):
            #     i = tf.stack([tf.range(3), a], 1)
            #     return tf.reduce_all(tf.gather_nd(action_mask, i))
            # def body(a, x):
            #     d = tfd.Categorical(x)
            #     a = d.sample()
            #     return (a, x)
            # action = tf.while_loop(cond, body, [action, logits])[0]
        return action

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    return dict(
        actor_encoder=Encoder(config['actor_encoder']), 
        actor_rnn=GRU(config['actor_gru']), 
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value_encoder=Encoder(config['value_encoder']),
        value_rnn=GRU(config['value_gru']),
        value=Value(config['value'])
    )


class PPO(Ensemble):
    def __init__(self, config, env, model_fn=create_components, **kwargs):
        super().__init__(
            model_fn=model_fn, 
            config=config,
            env=env,
            **kwargs)

    @tf.function
    def action(self, obs, shared_state, action_mask, state, mask, 
            evaluation=False, prev_action=None, prev_reward=None, **kwargs):
        assert obs.shape.ndims % 2 == 0, obs.shape
        actor_state, value_state = state
        x_actor, actor_state = self.encode(
            obs, actor_state, mask, 'actor', prev_action, prev_reward)
        act_dist = self.actor(x_actor, action_mask, evaluation=evaluation)
        action = self.actor.action(act_dist, action_mask, evaluation)

        if evaluation:
            return action, State(actor_state, value_state)
        else:
            x_value, value_state = self.encode(
                shared_state, actor_state, mask, 'value', prev_action, prev_reward)
            value = self.value(x_value)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            # intend to keep the batch dimension for later use
            out = (action, terms)
            return out, State(actor_state, value_state)

    @tf.function(experimental_relax_shapes=True)
    def compute_value(self, shared_state, state, mask, 
            prev_action=None, prev_reward=None, return_state=False):
        x, state = self.encode(
            shared_state, state, mask, 'value', prev_action, prev_reward)
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
        return x, state.h

    def _process_additional_input(self, x, prev_action, prev_reward):
        results = []
        if prev_action is not None:
            if self.actor.is_action_discrete:
                if prev_action.shape.ndims < 2:
                    prev_action = tf.reshape(prev_action, (-1, 1))
                prev_action = tf.one_hot(prev_action, self.actor.action_dim, dtype=x.dtype)
            else:
                if prev_action.shape.ndims < 3:
                    prev_action = tf.reshape(prev_action, (-1, 1, self.actor.action_dim))
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

    def reset_states(self, states=None):
        actor_state, value_state = states
        self.actor_rnn.reset_states(actor_state)
        self.value_rnn.reset_states(value_state)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        actor_state = self.actor_rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype) \
                if hasattr(self, 'actor_rnn') else None
        value_state = self.value_rnn.get_initial_state(
            inputs, batch_size=batch_size, dtype=dtype) \
                if hasattr(self, 'value_rnn') else None
        return State(actor_state.h, value_state.h)

    @property
    def state_size(self):
        return State(self.actor_rnn.state_size.h, self.value_rnn.state_size.h)
        
    @property
    def state_keys(self):
        return State(*State._fields)


def create_model(config, env, **kwargs):
    return PPO(config, env, create_components, **kwargs)
