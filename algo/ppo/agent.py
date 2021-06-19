import numpy as np
import tensorflow as tf

from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase, collect


class Agent(PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        dtype = tf.float32 if self._precision == 32 else tf.float16
        obs_dtype = dtype if np.issubdtype(env.obs_dtype, np.floating) else env.obs_dtype
        action_dtype = dtype if np.issubdtype(env.action_dtype, np.floating) else env.action_dtype
        TensorSpecs = dict(
            obs=(env.obs_shape, obs_dtype, 'obs'),
            action=(env.action_shape, action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)
