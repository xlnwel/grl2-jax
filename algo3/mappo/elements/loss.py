import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility.rl_loss import ppo_loss
from utility.tf_utils import explained_variance
from algo.ppo.elements.loss import PPOLossImpl


class MAPPOPolicyLoss(Loss):
    def loss(self, obs, action, advantage, logpi, 
            state=None, action_mask=None, 
            life_mask=None, mask=None):
        with tf.GradientTape() as tape:
            x_actor, _ = self.model.encode(obs, state, mask)
            act_dist = self.policy(x_actor, action_mask)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy, 
                life_mask if self._life_mask else None)
            loss = policy_loss - self._entropy_coef * entropy

        terms = dict(
            ratio=tf.exp(log_ratio),
            entropy=entropy,
            kl=kl,
            p_clip_frac=clip_frac,
            policy_loss=policy_loss,
            actor_loss=loss,
        )
        if action_mask is not None:
            terms['n_avail_actions'] = tf.reduce_sum(tf.cast(action_mask, tf.float32), -1)
        if life_mask is not None:
            terms['life_mask'] = life_mask

        return tape, loss, terms


class MAPPOValueLoss(PPOLossImpl):
    def loss(self, global_state, value, traj_ret, 
            state=None, life_mask=None, mask=None):
        old_value = value
        with tf.GradientTape() as tape:
            x_value, _ = self.model.encode(global_state, state, mask)
            value = self.value(x_value)

            loss, clip_frac = self._compute_value_loss(
                value, traj_ret, old_value,
                life_mask if self._life_mask else None)
            loss = self._value_coef * loss
        
        terms = dict(
            value=value,
            v_loss=loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=clip_frac,
        )

        return tape, loss, terms


def create_loss(config, model, name='mappo'):
    def constructor(config, cls, name):
        return cls(config=config, model=model[name], name=name)

    return LossEnsemble(
        config=config,
        model=model,
        constructor=constructor,
        name=name,
        policy=MAPPOPolicyLoss,
        value=MAPPOValueLoss,
    )
