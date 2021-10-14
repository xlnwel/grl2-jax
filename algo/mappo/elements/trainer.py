import tensorflow as tf

from core.elements.trainer import Trainer, TrainerEnsemble
from core.decorator import override
from core.tf_config import build


def get_data_format(config, env_stats, model, use_for_dataset=True):
    basic_shape = (None, config['sample_size'])
    data_format = dict(
        obs=((*basic_shape, *env_stats.obs_shape), env_stats.obs_dtype, 'obs'),
        global_state=((*basic_shape, *env_stats.global_state_shape), env_stats.global_state_dtype, 'global_state'),
        action=((*basic_shape, *env_stats.action_shape), env_stats.action_dtype, 'action'),
        value=(basic_shape, tf.float32, 'value'),
        traj_ret=(basic_shape, tf.float32, 'traj_ret'),
        advantage=(basic_shape, tf.float32, 'advantage'),
        logpi=(basic_shape, tf.float32, 'logpi'),
        mask=(basic_shape, tf.float32, 'mask'),
    )
    if env_stats.use_action_mask:
        data_format['action_mask'] = ((*basic_shape, env_stats.action_dim), tf.bool, 'action_mask')
    if env_stats.use_life_mask:
        data_format['life_mask'] = (basic_shape, tf.float32, 'life_mask')
    
    if config['store_state']:
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


class MAPPOActorTrainer(Trainer):
    def raw_train(self, obs, action, advantage, logpi, state=None, 
            action_mask=None, life_mask=None, mask=None):
        tape, loss, terms = self.loss.loss(
            obs, action, advantage, logpi, state, 
            action_mask, life_mask, mask)
        
        terms['actor_norm'] = self.optimizer(tape, loss)

        return terms


class MAPPOValueTrainer(Trainer):
    def raw_train(self, global_state, value, traj_ret, state=None, 
            life_mask=None, mask=None):
        tape, loss, terms = self.loss.loss(
            global_state, value, traj_ret, state, life_mask, mask)

        terms['value_norm'] = self.optimizer(tape, loss)

        return terms


class MAPPOTrainerEnsemble(TrainerEnsemble):
    @override(TrainerEnsemble)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.model, False)
        self.train = build(self.train, TensorSpecs)

    def raw_train(self, obs, global_state, action, 
            value, traj_ret, advantage, logpi, mask,
            action_mask=None, life_mask=None, state=None):
        actor_state, value_state = self.model.split_state(state)
        actor_terms = self.policy.raw_train(
            obs, action, advantage, logpi, actor_state, 
            action_mask, life_mask, mask
        )
        value_terms = self.value.raw_train(
            global_state, value, traj_ret, value_state,
            life_mask, mask
        )

        return {**actor_terms, **value_terms}

    @tf.function
    def train_value(self, global_state, value, 
            traj_ret, value_state, life_mask, mask):
        terms = self.value.raw_train(
            global_state, value, traj_ret, value_state,
            life_mask, mask
        )
        return terms

def create_trainer(config, model, loss, env_stats, name='mappo'):
    def constructor(config, cls, name):
        return cls(
            config=config, loss=loss[name], 
            env_stats=env_stats, name=name)

    return MAPPOTrainerEnsemble(
        config=config,
        model=model,
        loss=loss,
        env_stats=env_stats,
        constructor=constructor,
        name=name,
        policy=MAPPOActorTrainer,
        value=MAPPOValueTrainer,
    )
