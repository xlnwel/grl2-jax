import tensorflow as tf

from utility.rl_loss import ppo_loss
from utility.tf_utils import explained_variance
from algo.ppo.elements.loss import PPOLossImpl


class PPOLoss(PPOLossImpl):
    def loss(
        self, 
        obs, 
        action, 
        plogits, 
        paction, 
        value, 
        traj_ret, 
        advantage, 
        logpi, 
        state=None, 
        mask=None
    ):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x, pred_pact_dist, pred_paction, ppolicy_penultimate, _ = self.model.encode(obs, state, mask)
            pred_plogits = pred_pact_dist.logits
            pentropy = pred_pact_dist.entropy()
            pprob = tf.nn.softmax(plogits)
            pkl_loss = tf.nn.softmax_cross_entropy_with_logits(pprob, pred_plogits)
            pkl_loss = self.config.pa_coef * tf.reduce_mean(pkl_loss)
            psl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(paction, pred_plogits)
            psl_loss = tf.reduce_mean(psl_loss)
            pa_loss = self.config.pa_coef * pkl_loss

            # pa_one_hot = tf.one_hot(pred_paction, self.model.ppolicy.action_dim)
            # pa_one_hot = tf.stop_gradient(pa_one_hot - pred_plogits) + pred_plogits
            
            act_dist = self.policy(x, ppolicy_penultimate)
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
            loss = actor_loss + value_loss + pa_loss
        weights = self.model.policy.variables[0]
        encode_weights = weights[:-256]
        pa_weights = weights[-256:]

        ratio = tf.exp(log_ratio)
        terms.update(dict(
            value=value,
            ratio=ratio, 
            plogits=plogits,
            pentropy=pentropy,
            pkl_loss=pkl_loss,
            psl_loss=psl_loss,
            pa_loss=pa_loss,
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            policy_loss=policy_loss,
            actor_loss=actor_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac,
            encode_weights=encode_weights,
            pa_weights=pa_weights,
            paction_prec=tf.cast(paction == pred_paction, tf.float32)
        ))

        return tape, loss, terms

def create_loss(config, model, name='ppo'):
    return PPOLoss(config=config, model=model, name=name)
