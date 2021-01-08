import numpy as np
import tensorflow as tf

from utility.schedule import TFPiecewiseSchedule
from core.tf_config import build
from core.decorator import override
from core.optimizer import Optimizer
from algo.ppo.base import PPOBase


class Agent(PPOBase):
    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list)
            self._lr = TFPiecewiseSchedule(self._lr)
        models = [self.encoder, self.actor, self.value]
        if hasattr(self, 'rnn'):
            models.append(self.rnn)
        self._optimizer = Optimizer(
            self._optimizer, models, self._lr, 
            clip_norm=self._clip_norm, epsilon=self._opt_eps)

    @override(PPOBase)
    def _build_learn(self, env):
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

    @override(PPOBase)
    def _summary(self, data, terms):
        tf.summary.histogram('sum/value', data['value'], step=self._env_step)
        tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)
