import tensorflow as tf

from core.module import Ensemble
from algo.ppo.nn import Encoder, Actor, Value
from nn.func import LSTM
from utility.tf_utils import assert_rank


class PPO(Ensemble):
    def __init__(self, config, env, **kwargs):
        super().__init__(
            model_fn=create_components, 
            config=config,
            env=env,
            **kwargs)
    
    @tf.function
    def action(self, x, state, mask, evaluation=False, prev_action=None, prev_reward=None):
        assert x.shape.ndims % 2 == 0, x.shape
        x, state = self._encode(x, state, mask, 
            prev_action=prev_action, prev_reward=prev_reward)
        act_dist = self.actor(x, evaluation=evaluation)
        action = self.actor.action(act_dist, evaluation)
        if evaluation:
            action = tf.squeeze(action)
            return action, state
        else:
            value = self.value(x)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            # intend to keep the batch dimension for later use
            out = tf.nest.map_structure(lambda x: tf.squeeze(x, 1), (action, terms))
            return out, state

    @tf.function
    def compute_value(self, x, state, mask, prev_action=None, prev_reward=None):
        x, state = self._encode(x, state, mask, 
            prev_action=prev_action, prev_reward=prev_reward)
        value = self.value(x)
        value = tf.squeeze(value, 1)
        return value

    def reset_states(self, states=None):
        self.rnn.reset_states(states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.rnn.get_initial_state(inputs, batch_size=batch_size, dtype=dtype)

    def _encode(self, x, state, mask, prev_action=None, prev_reward=None):
        x = tf.expand_dims(x, 1)
        mask = tf.expand_dims(mask, 1)
        x = self.encoder(x)
        additional_rnn_input = self._process_additional_input(
            x, prev_action, prev_reward)
        x, state = self.rnn(x, state, mask, 
            additional_rnn_input=additional_rnn_input)
        return x, state

    def _process_additional_input(self, x, prev_action, prev_reward):
        results = []
        if self.additional_rnn_input:
            if prev_action is not None:
                if self.actor.is_action_discrete:
                    prev_action = tf.reshape(prev_action, (-1, 1))
                    prev_action = tf.one_hot(prev_action, self.actor.action_dim, dtype=x.dtype)
                else:
                    prev_action = tf.reshape(prev_action, (-1, 1, self.actor.action_dim))
                results.append(prev_action)
            if prev_reward is not None:
                prev_reward = tf.reshape(prev_reward, (-1, 1, 1))
                results.append(prev_reward)
        [assert_rank(v, 3) for v in results]
        return results

    @property
    def state_size(self):
        return self.rnn.state_size
    
    @property
    def state_keys(self):
        return self.rnn.state_keys

def create_components(config, env):
    action_dim = env.action_dim
    is_action_discrete = env.is_action_discrete

    config['encoder']['time_distributed'] = True
    return dict(
        encoder=Encoder(config['encoder']), 
        rnn=LSTM(config['rnn']),
        actor=Actor(config['actor'], action_dim, is_action_discrete),
        value=Value(config['value'])
    )

def create_model(config, env, **kwargs):
    return PPO(config, env, **kwargs)
