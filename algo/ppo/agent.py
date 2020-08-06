import cloudpickle
import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import agent_config
from algo.ppo.base import PPOBase
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(PPOBase):
    @agent_config
    def __init__(self, dataset, env):
        super().__init__(dataset=dataset, env=env)

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            traj_ret=((), tf.float32, 'traj_ret'),
            value=((), tf.float32, 'value'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    def reset_states(self, states=None):
        pass

    def get_states(self):
        return None

    def __call__(self, obs, deterministic=False, update_rms=False, **kwargs):
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)
        if update_rms:
            self.update_obs_rms(obs)
        obs = self.normalize_obs(obs)
        if deterministic:
            return self.model.action(obs, deterministic).numpy()
        else:
            out = self.model.action(obs, deterministic)
            action, terms = tf.nest.map_structure(lambda x: x.numpy(), out)
            terms['obs'] = obs  # return normalized obs 
            return action, terms
