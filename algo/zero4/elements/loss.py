import tensorflow as tf

from utility.rl_loss import ppo_loss
from utility.tf_utils import explained_variance
from algo.ppo.elements.loss import PPOLossImpl


class PPOLoss(PPOLossImpl):
    def loss(self, obs, goal, action, value, traj_ret, advantage, logpi, 
                state=None, mask=None):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x, state = self.model.encode(obs, state, mask)
            goal = self.goal_encoder(goal)
            act_dist = self.policy(x, goal)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self.config.clip_range, entropy)
            # value loss
            value = self.value(x)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)

            actor_loss = (policy_loss - self.config.entropy_coef * entropy)
            value_loss = self.config.value_coef * value_loss
            loss = actor_loss + value_loss

        ratio = tf.exp(log_ratio)
        terms.update(dict(
            value=value,
            ratio=ratio, 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            policy_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return tape, loss, terms

def create_loss(config, model, name='mappo'):
    return PPOLoss(config=config, model=model, name=name)
