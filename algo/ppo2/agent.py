import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from core.base import Memory
from algo.ppo.base import PPOBase


class Agent(Memory, PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        PPOBase._add_attributes(self, env, dataset)
        Memory._add_attributes(self)

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
        obs, kwargs = PPOBase._process_input(self, obs, evaluation, env_output)
        obs, kwargs = Memory._process_input(self, obs, env_output, kwargs)
        return obs, kwargs

    # @override(PPOBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = Memory._process_output(self, obs, kwargs, out, evaluation)
        out = PPOBase._process_output(self, obs, kwargs, out, evaluation)
        if not evaluation:
            out[1]['mask'] = kwargs['mask']
        return out

    """ PPO methods """
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
