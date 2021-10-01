import numpy as np
import tensorflow as tf

from core.decorator import override
from core.module import Trainer
from core.tf_config import build


class PPOTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        dtype = tf.float32
        obs_dtype = dtype if np.issubdtype(env_stats.obs_dtype, np.floating) else env_stats.obs_dtype
        action_dtype = dtype if np.issubdtype(env_stats.action_dtype, np.floating) else env_stats.action_dtype
        TensorSpecs = dict(
            obs=(env_stats.obs_shape, obs_dtype, 'obs'),
            action=(env_stats.action_shape, action_dtype, 'action'),
            value=((), tf.float32, 'value'),
            traj_ret=((), tf.float32, 'traj_ret'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        if hasattr(self.model, 'rnn'):
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        self.train = build(self.train, TensorSpecs)

    def raw_train(self, obs, action, value, traj_ret, 
            advantage, logpi, state=None, mask=None):
        tape, loss, terms = self.loss.loss(
            obs, action, value, traj_ret, 
            advantage, logpi, state, mask)
        terms['ppo_norm'] = self.optimizer(tape, loss)

        return terms


def create_trainer(config, model, loss, env_stats, name='ppo'):
    trainer = PPOTrainer(
        config=config, model=model, loss=loss, 
        env_stats=env_stats, name=name)

    return trainer
