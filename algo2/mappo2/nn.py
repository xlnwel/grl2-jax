import tensorflow as tf

from utility.tf_utils import assert_rank
from algo2.mappo.nn import PPO as PPOBase, create_components


class PPO(PPOBase):
    @tf.function
    def action(self, obs, shared_state, action_mask, state, mask, 
            evaluation=False, prev_action=None, prev_reward=None, **kwargs):
        assert obs.shape.ndims % 2 != 0, obs.shape

        mid = len(state) // 2
        actor_state, value_state = state[:mid], state[mid:]
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
                shared_state, value_state, mask, 'value', 
                prev_action, prev_reward)
            value = self.value(x_value)
            logpi = act_dist.log_prob(action)
            terms = {'logpi': logpi, 'value': value}
            out = (action, terms)
            return out, self.State(*actor_state, *value_state)

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
        if x.shape.ndims % 2 != 0:
            x = tf.expand_dims(x, 1)
        if mask.shape.ndims < 3:
            mask = tf.reshape(mask, (-1, 1, mask.shape[-1]))
        assert_rank(mask, 3)
        assert_rank(x, 4)

        x = encoder(x)                          # [B, S, A, F]
        seqlen, n_agents = x.shape[1:3]
        if self.pool_communication:
            pool = tf.reduce_max(x, axis=2, keepdims=True)
            n = x.shape[-1]
            i = tf.reshape(tf.range(n, dtype=tf.int32), (1, 1, 1, -1))
            x = tf.where(i < tf.cast(n * self.pool_frac, tf.int32), pool, x)
        x = tf.transpose(x, [0, 2, 1, 3])       # [B, A, S, F]
        x = tf.reshape(x, [-1, *x.shape[2:]])   # [B * A, S, F]
        mask = tf.transpose(mask, [0, 2, 1])    # [B, A, S]
        mask = tf.reshape(mask, [-1, mask.shape[-1]])   # [B * A, S]
        additional_rnn_input = self._process_additional_input(
            x, prev_action, prev_reward)
        x, state = rnn(x, state, mask, 
            additional_input=additional_rnn_input)
        x = tf.reshape(x, (-1, n_agents, seqlen, x.shape[-1]))  # [B, A, S, F]
        x = tf.transpose(x, [0, 2, 1, 3])       # [B, S, A, F]

        if seqlen == 1:
            x = tf.squeeze(x, 1)
        
        return x, state

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
