import tensorflow as tf

from utility.rl_loss import reduce_mean, ppo_loss
from utility.tf_utils import explained_variance
from algo.hm.elements.loss import PPOLossImpl


class PPOLoss(PPOLossImpl):
    def loss(
        self, 
        obs, 
        action, 
        value, 
        traj_ret, 
        advantage, 
        logprob, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        old_value = value
        loss_mask = life_mask
        n = None if loss_mask is None else tf.reduce_sum(loss_mask)
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                state=state, 
                mask=mask
            )
            act_dist = self.policy(x)
            new_logprob = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logprob - logprob
            raw_policy_loss, raw_entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, 
                advantage, 
                self.config.clip_range, 
                entropy, 
                reduce=False
            )
            policy_loss = reduce_mean(raw_policy_loss, loss_mask, n)
            entropy = reduce_mean(raw_entropy, loss_mask, n)
            entropy_loss = - self.config.entropy_coef * entropy
            actor_loss = policy_loss + entropy_loss
            # value loss
            value = self.value(x)
            raw_v_loss, v_clip_frac = self._compute_value_loss(
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=loss_mask,
                reduce=False
            )
            value_loss = reduce_mean(raw_v_loss, loss_mask, n)
            value_loss = self.config.value_coef * value_loss
            loss = actor_loss + value_loss

        prob = tf.exp(logprob)
        new_prob = tf.exp(new_logprob)
        ratio = tf.exp(log_ratio)
        terms = dict(
            value=value,
            ratio=ratio, 
            entropy=entropy, 
            kl=kl, 
            logprob=logprob,
            new_logprob=new_logprob, 
            prob=prob,
            new_prob=new_prob, 
            diff_prob=new_prob - prob, 
            p_clip_frac=p_clip_frac,
            raw_policy_loss=raw_policy_loss,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss, 
            actor_loss=actor_loss,
            raw_v_loss=raw_v_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        )
        if action_mask is not None:
            terms['n_avail_actions'] = tf.reduce_sum(
                tf.cast(action_mask, tf.float32), -1)
        if not self.policy.is_action_discrete:
            terms['policy_mean'] = act_dist.mean()
            terms['policy_std'] = tf.exp(self.policy.logstd)

        return tape, loss, terms

def create_loss(config, model, name='ppo'):
    return PPOLoss(config=config, model=model, name=name)
