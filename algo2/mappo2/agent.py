import numpy as np

from core.decorator import override
from algo.ppo.base import PPOBase
from algo2.mappo.agent import Agent as AgentBase, infer_life_mask


def collect(buffer, env, step, reset, life_mask, discount, next_obs, **kwargs):
    kwargs['life_mask'] = infer_life_mask(life_mask, discount, concat=False)
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = discount
    buffer.add(**kwargs)

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    a = env.random_action()
    terms = {
        'obs': obs['obs'], 
        'global_state': obs['global_state'],
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
    def _reshape_env_output(self, env_output):
        return env_output

    # @override(PPOBase)
    def _process_input(self, env_output, evaluation):
        if evaluation:
            self._process_obs(env_output.obs, update_rms=False)
        else:
            life_mask = env_output.obs['life_mask']
            self._process_obs(env_output.obs, mask=life_mask)
        mask = self._get_mask(env_output.reset)
        mask_flat = np.concatenate(mask)
        obs, kwargs = self._divide_obs(env_output.obs)
        kwargs = self._add_memory_state_to_kwargs(
            obs, mask=mask_flat, kwargs=kwargs, 
            batch_size=obs.shape[0] * self._n_agents)
        kwargs['mask'] = mask
        
        return obs, kwargs

    """ PPO methods """
    def record_last_env_output(self, env_output):
        self._env_output = self._reshape_env_output(env_output)
        self._process_obs(self._env_output.obs, update_rms=False)
        mask = self._get_mask(self._env_output.reset)
        mask = np.concatenate(mask)
        self._state = self._apply_mask_to_state(self._state, mask)
