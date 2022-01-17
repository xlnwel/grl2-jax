from tensorflow_probability import distributions as tfd

from algo.ppo.elements.loss import *


class PPGLoss(PPOLoss):
    def aux_loss(
        self, 
        obs, 
        value, 
        traj_ret, 
        logits, 
        state=None,
        mask=None
    ):
        old_value = value
        terms = {}
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(obs, state, mask)
            # bc loss
            act_dist = self.policy(x)
            old_act_dist = tfd.Categorical(logits)
            kl = tf.reduce_mean(old_act_dist.kl_divergence(act_dist))
            bc_loss = self.config.bc_coef * kl
            # value loss
            value = self.value(x)
            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value)

            loss = bc_loss + value_loss

        terms.update(dict(
            value=value,
            kl=kl,
            bc_loss=bc_loss,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac
        ))

        return tape, loss, terms

def create_loss(config, model, name='mappo'):
    Loss = PPOLoss if config.get('type', 'mappo') else PPGLoss
    return Loss(config=config, model=model, name=name)
