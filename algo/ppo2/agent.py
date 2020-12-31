import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import agent_config
from nn.rnn import LSTMState
from algo.ppo.base import PPOBase
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(PPOBase):
    @agent_config
    def __init__(self, *, dataset, env):
        super().__init__(dataset=dataset, env=env)

        # previous and current state of LSTM
        self._state = self.model.get_initial_state(batch_size=env.n_envs)
        self._prev_action = None
        self._action_dim = env.action_dim
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size, *env.obs_shape), env.obs_dtype, 'obs'),
            action=((self._sample_size, *env.action_shape), env.action_dtype, 'action'),
            value=((self._sample_size,), tf.float32, 'value'),
            traj_ret=((self._sample_size,), tf.float32, 'traj_ret'),
            advantage=((self._sample_size,), tf.float32, 'advantage'),
            logpi=((self._sample_size,), tf.float32, 'logpi'),
            mask=((self._sample_size,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        self.learn = build(self._learn, TensorSpecs, print_terminal_info=True)

    def reset_states(self, states=None):
        if states is None:
            self._state, self._prev_action, self._prev_reward = None, None, None
        else:
            self._state, self._prev_action, self._prev_reward= states

    def get_states(self):
        return self._state, self._prev_action

    def __call__(self, obs, reset=None, evaluation=False, 
                update_curr_state=True, **kwargs):
        if len(obs.shape) % 2 != 0:
            obs = np.expand_dims(obs, 0)    # add batch dimension
        assert len(obs.shape) in (2, 4), obs.shape  
        if self._state is None:
            self._state = self.model.get_initial_state(batch_size=tf.shape(obs)[0])
            self._prev_action = tf.zeros(tf.shape(obs)[0], dtype=tf.int32)
        if reset is None:
            mask = tf.ones(tf.shape(obs)[0], dtype=tf.float32)
        else:
            mask = tf.cast(1. - reset, tf.float32)
        obs = self.normalize_obs(obs)
        prev_state = self._state
        out, state = self.model.action(obs, self._state, mask, self._deterministic_evaluation)
        if update_curr_state:
            self._state = state
        if evaluation:
            return out.numpy()
        else:
            action, terms = out
            terms['mask'] = mask
            if self._store_state:
                terms.update(prev_state._asdict())
            terms = tf.nest.map_structure(lambda x: x.numpy(), terms)
            return action.numpy(), terms
