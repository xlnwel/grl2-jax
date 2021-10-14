import tensorflow as tf

from core.elements.trainer import Trainer
from core.decorator import override
from core.tf_config import build


def get_data_format(config, env_stats, model, use_for_dataset=True):
    basic_shape = (None, config['sample_size']) \
        if hasattr(model, 'rnn') else (None,)
    data_format = dict(
        obs=((*basic_shape, *env_stats.obs_shape), env_stats.obs_dtype, 'obs'),
        action=((*basic_shape, *env_stats.action_shape), env_stats.action_dtype, 'action'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logpi=(basic_shape, tf.float32, 'logpi'),
    )

    if config.get('store_state'):
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        if use_for_dataset:
            data_format.update({
                name: ((None, sz), dtype)
                    for name, sz in model.state_size._asdict().items()
            })
        else:
            state_type = type(model.state_size)
            data_format['state'] = state_type(*[((None, sz), dtype, name) 
                for name, sz in model.state_size._asdict().items()])
    
    return data_format


class PPOTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.loss.model, False)
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
