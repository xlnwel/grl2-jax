import tensorflow as tf

from core.elements.trainer import Trainer, TrainerEnsemble
from core.decorator import override
from core.tf_config import build
from .utils import get_data_format


class PPOActorTrainer(Trainer):
    def raw_train(
        self, 
        **kwargs
    ):
        if not hasattr(self.model, 'rnn') \
                and len(kwargs['action'].shape) == 3:
            new_kwargs = tf.nest.map_structure(
                lambda x: tf.reshape(x, (-1, *x.shape[2:])) 
                if x is not None else x, kwargs)
        else:
            new_kwargs = kwargs
        tape, loss, terms = self.loss.loss(**new_kwargs)

        terms['actor_norm'], terms['actor_var_norm'] = \
            self.optimizer(tape, loss, return_var_norms=True)

        return terms


class PPOValueTrainer(Trainer):
    def raw_train(
        self, 
        **kwargs, 
    ):
        tape, loss, terms = self.loss.loss(
            **kwargs
        )

        terms['value_norm'], terms['value_var_norm'] = \
            self.optimizer(tape, loss, return_var_norms=True)

        return terms


class PPOTrainerEnsemble(TrainerEnsemble):
    @override(TrainerEnsemble)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.model)
        self.train = build(self.train, TensorSpecs)
        return True

    def raw_train(
        self, 
        obs, 
        action, 
        value, 
        traj_ret, 
        advantage, 
        logprob, 
        global_state=None, 
        reward=None, 
        raw_adv=None, 
        raw_aux_adv=None, 
        prev_reward=None, 
        prev_action=None, 
        action_mask=None, 
        life_mask=None, 
        actor_state=None, 
        value_state=None, 
        mask=None
    ):
        actor_terms = self.policy.raw_train(
            obs=obs, 
            action=action, 
            advantage=advantage, 
            logprob=logprob, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=actor_state, 
            action_mask=action_mask, 
            life_mask=life_mask, 
            mask=mask
        )
        global_state = obs if global_state is None else global_state
        value_terms = self.value.raw_train(
            global_state=global_state, 
            value=value, 
            traj_ret=traj_ret, 
            prev_reward=prev_reward, 
            prev_action=prev_action, 
            state=value_state, 
            life_mask=life_mask,
            mask=mask
        )

        return {**actor_terms, **value_terms}


def create_trainer(config, env_stats, loss, name='ppo'):
    def constructor(config, env_stats, cls, name):
        return cls(
            config=config, 
            env_stats=env_stats, 
            loss=loss[name], 
            name=name)

    return PPOTrainerEnsemble(
        config=config,
        env_stats=env_stats,
        loss=loss,
        constructor=constructor,
        name=name,
        policy=PPOActorTrainer,
        value=PPOValueTrainer,
    )
