import tensorflow as tf

from core.elements.trainer import Trainer, TrainerEnsemble
from core.decorator import override
from core.optimizer import create_optimizer
from core.tf_config import build
from utility.utils import dict2AttrDict
from .utils import get_data_format
from utility.adam import Adam
from utility.display import print_dict


class GPOActorTrainer(Trainer):
    def construct_optimizers(self):
        keys = sorted([
            k for k in self.model.keys() if not k.startswith('target')])
        modules = tuple([
            self.model[k] for k in keys if not k.startswith('meta')
        ])
        config = dict2AttrDict(self.config, to_copy=True)
        config.optimizer.opt_name = Adam
        self.optimizer = create_optimizer(
            modules, config.optimizer
        )
        
        if self.config.get('meta_opt'):
            meta_modules = tuple([
                self.model[k] for k in keys if k.startswith('meta')
            ])
            self.meta_opt = create_optimizer(
                meta_modules, self.config.meta_opt
            )

    def raw_train(
        self, 
        obs, 
        action, 
        advantage, 
        logprob, 
        target_prob, 
        tr_prob, 
        target_prob_prime, 
        tr_prob_prime, 
        pi, 
        target_pi, 
        pi_mean, 
        pi_std, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None, 
    ):
        # if not hasattr(self.model, 'rnn') \
        #         and len(kwargs['action'].shape) == 3:
        #     new_kwargs = tf.nest.map_structure(
        #         lambda x: tf.reshape(x, (-1, *x.shape[2:])) 
        #         if x is not None else x, kwargs)
        # else:
        #     new_kwargs = kwargs
        loss_mask = life_mask if self.loss.config.life_mask else None
        n = None if loss_mask is None else tf.reduce_sum(loss_mask)
        with tf.GradientTape(persistent=True) as meta_tape:
            tape, loss, terms = self.loss.loss(
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
                state=state, 
                action_mask=action_mask, 
                life_mask=life_mask, 
                mask=mask, 
                use_meta=True, 
            )

            terms['actor_norm'], terms['actor_var_norm'] = \
                self.optimizer(tape, loss, return_var_norms=True)
            
            x, _ = self.model.encode(
                x=obs, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            act_dist = self.loss.policy(x, action_mask)
            meta_terms, meta_loss = self.loss._ppo_loss(
                tape=meta_tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                logprob=logprob, 
                tr_prob=tr_prob, 
                target_prob_prime=target_prob_prime, 
                tr_prob_prime=tr_prob_prime, 
                pi=pi, 
                target_pi=target_pi, 
                pi_mean=pi_mean, 
                pi_std=pi_std, 
                action_mask=action_mask, 
                sample_mask=loss_mask, 
                n=n, 
                use_meta=False, 
                name='meta'
            )
        out_grads = meta_tape.gradient(
            meta_loss, 
            self.optimizer.variables
        )
        out_grads = meta_tape.gradient(
            self.optimizer.get_transformed_grads(self.optimizer.variables), 
            self.optimizer.grads, 
            output_gradients=out_grads
        )
        terms.update(meta_terms)
        terms['meta_norm'], terms['meta_var_norm'] = \
            self.meta_opt(
                meta_tape, 
                self.optimizer.grads, 
                output_gradients=out_grads, 
                return_var_norms=True
            )

        return terms


class GPOValueTrainer(Trainer):
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


class GPOTrainerEnsemble(TrainerEnsemble):
    @override(TrainerEnsemble)
    def _build_train(self, env_stats):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.model)
        print_dict(TensorSpecs, prefix='Tensor Specifications')
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
