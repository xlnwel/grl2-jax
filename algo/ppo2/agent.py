from core.base import RMSBaseAgent
import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        # previous and current state of LSTM
        self._state = None
        self._prev_action = None
    
    @override(PPOBase)
    def _build_learn(self, env):
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
        if self.model.additional_rnn_input:
            TensorSpecs['additional_rnn_input'] = [(
                ((self._sample_size, env.action_dim), self._dtype, 'prev_action'),
                ((self._sample_size, 1), self._dtype, 'reward'),    # this reward should be unnormlaized
            )]
        self.learn = build(self._learn, TensorSpecs, print_terminal_info=True)

    @override(PPOBase)
    def reset_states(self, states=None):
        if states is None:
            self._state, self._prev_action = None, None
        else:
            self._state, self._prev_action = states

    @override(PPOBase)
    def get_states(self):
        return self._state, self._prev_action

    def record_last_obs(self, env_output):
        self.update_obs_rms(env_output.obs)
        self._last_obs = self.normalize_obs(env_output.obs)
        self._mask = 1 - env_output.reset

    def compute_value(self, obs=None, state=None, mask=None, prev_action=None, reward=None, return_state=False):
        # be sure you normalize obs first if obs normalization is required
        obs = obs or self._last_obs
        mask = mask or self._mask
        state = state or self._state
        prev_action = prev_action or self._prev_action
        reward = reward or self._reward
        out = self.model.compute_value(self._last_obs,
            state, mask, prev_action, reward)
        if return_state:
            return tf.nest.map_structure(lambda x: x.numpy(), out)
        else:
            return out[0].numpy()

    @override(RMSBaseAgent)
    def _process_input(self, obs, evaluation, env_output):
        super()._process_input(obs, evaluation, env_output)
        self._reward = env_output.reward # use unnormalized reward to avoid potential inconsistency

        if self._state is None:
            self._state = self.model.get_initial_state(batch_size=tf.shape(obs)[0])
            if self.model.additional_rnn_input:
                self._prev_action = np.zeros(obs.shape[0], dtype=np.int32)

        mask = 1. - reset   # mask is applied in LSTM
        prev_state = self._state
        out, self._state = self.model.action(obs, self._state, mask, evaluation,
            prev_action=self._prev_action, reward=self._reward)
        out = tf.nest.map_structure(lambda x: x.numpy(), out)
        if not evaluation:
            terms = out[1]
            if self._store_state:
                terms.update(tf.nest.map_structure(
                    lambda x: x.numpy(), prev_state._asdict()))
            terms['mask'] = mask
            terms['obs'] = obs
            if self.model.additional_rnn_input:
                terms['prev_action'] = self._prev_action
        if self.model.additional_rnn_input:
            self._prev_action = out[0] if isinstance(out, tuple) else out
        return out
