import tensorflow as tf

from core.decorator import override
from core.module import Trainer, TrainerEnsemble
from core.tf_config import build


class MAPPOActorTrainer(Trainer):
    def raw_train(self, obs, action, advantage, logpi, state=None, 
            action_mask=None, life_mask=None, mask=None):
        print('retracing train actor')
        tape, loss, terms = self.loss.loss(
            obs, action, advantage, logpi, state, 
            action_mask, life_mask, mask)
        
        terms['actor_norm'] = self.optimizer(tape, loss)

        return terms


class MAPPOValueTrainer(Trainer):
    def raw_train(self, global_state, value, traj_ret, state=None, 
            life_mask=None, mask=None):
        print('retracing train value function')
        tape, loss, terms = self.loss.loss(
            global_state, value, traj_ret, state, life_mask, mask)

        terms['value_norm'] = self.optimizer(tape, loss)

        return terms


class MAPPOTrainerEnsemble(TrainerEnsemble):
    @override(Trainer)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        basic_shape = (self._sample_size,)
        TensorSpecs = dict(
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
            TensorSpecs['action_mask'] = ((*basic_shape, env_stats.action_dim), tf.bool, 'action_mask')
        if env_stats.use_life_mask:
            TensorSpecs['life_mask'] = (basic_shape, tf.float32, 'life_mask')
        
        if self._store_state:
            dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        self.train = build(self.train, TensorSpecs)

    def raw_train(self, obs, global_state, action, 
            value, traj_ret, advantage, logpi, mask,
            action_mask=None, life_mask=None, state=None):
        print('retracing PPOTrainerEnsemble train')
        actor_state, value_state = self.model.split_state(state)
        actor_terms = self.actor.raw_train(
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
            config=config, model=model[name], loss=loss[name], 
            env_stats=env_stats, name=name)

    return MAPPOTrainerEnsemble(
        config=config,
        model=model,
        loss=loss,
        env_stats=env_stats,
        constructor=constructor,
        name=name,
        actor=MAPPOActorTrainer,
        value=MAPPOValueTrainer,
    )
