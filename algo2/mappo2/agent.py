import numpy as np

from core.decorator import override
from algo.ppo.base import PPOBase
from algo2.mappo.agent import Agent as AgentBase


def collect(buffer, env, step, reset, discount, next_obs, **kwargs):
    kwargs['life_mask'] = discount.copy()
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = discount
    buffer.add(**kwargs)

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    a = env.random_action()
    terms = {
        'obs': obs['obs'], 
        'shared_state': obs['shared_state'],
    }
    return a, terms

class Agent(AgentBase):    
    """ PPO methods """
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._basic_shape = (self._sample_size, self._n_agents)

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
