import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import agent_config
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    @agent_config
    def __init__(self, *, dataset, env):
        super().__init__(dataset=dataset, env=env)

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    def __call__(self, obs, evaluation=False, env_output=None, **kwargs):
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)    # add batch dimension
        if not evaluation:
            self.update_obs_rms(obs)
            self.update_reward_rms(env_output.reward, env_output.discount)
        obs = self.normalize_obs(obs)
        out = tf.nest.map_structure(
            lambda x: x.numpy(), self.model.action(obs, evaluation))
        
        if not evaluation:
            terms = out[1]
            terms['obs'] = obs
            terms['reward'] = self.normalize_reward(env_output.reward)
        return out
