import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        # previous and current state of LSTM
        self._state = None
        self._prev_action = 0
        self._prev_reward = 0

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
                ((self._sample_size, 1), self._dtype, 'prev_reward'),    # this reward should be unnormlaized
            )]
        self.learn = build(self._learn, TensorSpecs, print_terminal_info=True)

    """ Call """
    # @override(PPOBase)
    def _process_input(self, obs, evaluation, env_output):
        if self._state is None:
            self._state = self.model.get_initial_state(batch_size=tf.shape(obs)[0])
            if self.model.additional_rnn_input:
                self._prev_action = tf.zeros(obs.shape[0], dtype=tf.int32)
        
        obs, kwargs = super()._process_input(obs, evaluation, env_output)
        kwargs.update({
            'state': self._state,
            'mask': 1. - env_output.reset,   # mask is applied in LSTM
            'prev_action': self._prev_action, 
            'prev_reward': env_output.reward # use unnormalized reward to avoid potential inconsistency
        })
        return obs, kwargs
        
    # @override(PPOBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out, self._state = out
        if self.model.additional_rnn_input:
            self._prev_action = out[0]
        
        out = super()._process_output(obs, kwargs, out, evaluation)
        if not evaluation:
            terms = out[1]
            if self._store_state:
                terms.update(tf.nest.map_structure(
                    lambda x: x.numpy(), kwargs['state']._asdict()))
            terms.update({
                'obs': obs,
                'mask': kwargs['mask'],
            })
        return out

    """ PPO methods """
    @override(PPOBase)
    def reset_states(self, states=None):
        if states is None:
            self._state, self._prev_action = None, None
        else:
            self._state, self._prev_action = states

    @override(PPOBase)
    def get_states(self):
        return self._state, self._prev_action

    @override(PPOBase)
    def record_last_env_output(self, env_output):
        self.update_obs_rms(env_output.obs)
        self._last_obs = self.normalize_obs(env_output.obs)
        self._mask = 1 - env_output.reset
        self._prev_reward = env_output.reward

    @override(PPOBase)
    def compute_value(self, obs=None, state=None, mask=None, 
                    prev_action=None, prev_reward=None, return_state=False):
        # be sure obs is normalized if obs normalization is required
        obs = obs or self._last_obs
        mask = mask or self._mask
        state = state or self._state
        prev_action = prev_action or self._prev_action
        prev_reward = prev_reward or self._prev_reward
        out = self.model.compute_value(
            obs, state, mask, prev_action, prev_reward)
        if return_state:
            return tf.nest.map_structure(lambda x: x.numpy(), out)
        else:
            return out[0].numpy()
