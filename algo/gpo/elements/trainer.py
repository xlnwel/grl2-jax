import tensorflow as tf

from core.elements.trainer import Trainer, TrainerEnsemble
from core.decorator import override
from core.optimizer import create_optimizer
from core.tf_config import build
from .utils import get_data_format
from utility.display import print_dict


class GPOActorTrainer(Trainer):
    def construct_optimizers(self):
        keys = sorted([
            k for k in self.model.keys() if not k.startswith('target')])
        modules = tuple([
            self.model[k] for k in keys if not k.startswith('meta')
        ])
        self.optimizer = create_optimizer(
            modules, self.config.optimizer
        )

    def raw_train(
        self, 
        **kwargs
    ):
        # if not hasattr(self.model, 'rnn') \
        #         and len(kwargs['action'].shape) == 3:
        #     new_kwargs = tf.nest.map_structure(
        #         lambda x: tf.reshape(x, (-1, *x.shape[2:])) 
        #         if x is not None else x, kwargs)
        # else:
        #     new_kwargs = kwargs
        tape, loss, terms = self.loss.loss(**kwargs)

        terms['actor_norm'], var_norms = \
            self.optimizer(tape, loss, return_var_norms=True)
        terms['actor_var_norm'] = list(var_norms.values())

        return terms


class GPOValueTrainer(Trainer):
    def raw_train(
        self, 
        **kwargs, 
    ):
        tape, loss, terms = self.loss.loss(
            **kwargs
        )

        terms['value_norm'], var_norms = \
            self.optimizer(tape, loss, return_var_norms=True)
        terms['value_var_norm'] = list(var_norms.values())

        return terms


class GPOTrainerEnsemble(TrainerEnsemble):
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
        target_prob, 
        tr_prob, 
        target_prob_prime, 
        tr_prob_prime, 
        pi=None, 
        target_pi=None, 
        pi_mean=None, 
        pi_std=None, 
        global_state=None, 
        reward=None, 
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
            target_prob=target_prob, 
            tr_prob=tr_prob, 
            target_prob_prime=target_prob_prime, 
            tr_prob_prime=tr_prob_prime, 
            pi=pi,
            target_pi=target_pi, 
            pi_mean=pi_mean, 
            pi_std=pi_std, 
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

        return {**actor_terms, **value_terms, 
            'reward_frequency': tf.reduce_mean(tf.cast(reward != 0, tf.float32))}


def create_trainer(config, env_stats, loss, name='ppo'):
    def constructor(config, env_stats, cls, name):
        return cls(
            config=config, 
            env_stats=env_stats, 
            loss=loss[name], 
            name=name)

    return GPOTrainerEnsemble(
        config=config,
        env_stats=env_stats,
        loss=loss,
        constructor=constructor,
        name=name,
        policy=GPOActorTrainer,
        value=GPOValueTrainer,
    )
