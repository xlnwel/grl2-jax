import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase
from algo2.mappo.agent import Agent as AgentBase


def collect(buffer, env, step, reset, discount, next_obs, **kwargs):
    kwargs['life_mask'] = discount.copy()
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = discount
    buffer.add(**kwargs)

class Agent(AgentBase):    
    """ PPO methods """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size, self._n_agents, *env.obs_shape), env.obs_dtype, 'obs'),
            shared_state=((self._sample_size, self._n_agents, *env.shared_state_shape), env.shared_state_dtype, 'shared_state'),
            action_mask=((self._sample_size, self._n_agents, env.action_dim), tf.bool, 'action_mask'),
            action=((self._sample_size, self._n_agents, *env.action_shape), env.action_dtype, 'action'),
            value=((self._sample_size, self._n_agents), tf.float32, 'value'),
            traj_ret=((self._sample_size, self._n_agents), tf.float32, 'traj_ret'),
            advantage=((self._sample_size, self._n_agents), tf.float32, 'advantage'),
            logpi=((self._sample_size, self._n_agents), tf.float32, 'logpi'),
            life_mask=((self._sample_size, self._n_agents), tf.float32, 'life_mask'),
            mask=((self._sample_size, self._n_agents), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = ((self._sample_size, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = ((self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs)

        TensorSpecs = dict(
            shared_state=((self._sample_size, self._n_agents, *env.shared_state_shape), env.shared_state_dtype, 'shared_state'),
            value=((self._sample_size, self._n_agents), tf.float32, 'value'),
            traj_ret=((self._sample_size, self._n_agents), tf.float32, 'traj_ret'),
            life_mask=((self._sample_size, self._n_agents), tf.float32, 'life_mask'),
            mask=((self._sample_size, self._n_agents), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.value_state_size)
            TensorSpecs['value_state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.value_state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = ((self._sample_size, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = ((self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        
        learn_value = tf.function(self._learn_value)
        self.learn_value = build(learn_value, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    """ Call """
    # @override(PPOBase)
    def _reshape_output(self, env_output):
        return env_output

    # @override(PPOBase)
    def _process_input(self, env_output, evaluation):
        if evaluation:
            self._process_obs(env_output.obs, update_rms=False)
        else:
            life_mask = env_output.discount
            self._process_obs(env_output.obs, mask=life_mask)
        mask = 1. - env_output.reset
        obs, kwargs = self._divide_obs(env_output.obs)
        obs, kwargs = self._add_memory_state_to_kwargs(
            obs, mask=mask, kwargs=kwargs, batch_size=obs.shape[0] * self._n_agents)
        return obs, kwargs
