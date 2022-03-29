import tensorflow as tf

from core.elements.loss import LossEnsemble
from utility.tf_utils import explained_variance
from algo.hm.elements.loss import PPOLossImpl, MAPPOPolicyLoss


class MAPPOValueLoss(PPOLossImpl):
    def loss(
        self, 
        global_state, 
        action, 
        value, 
        value_a, 
        traj_ret, 
        traj_ret_a, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        life_mask=None, 
        mask=None
    ):
        old_value = value
        old_value_a = value_a
        with tf.GradientTape() as tape:
            value, value_a, _ = self.model.compute_value(
                global_state=global_state,
                action=action, 
                prev_reward=prev_reward,
                prev_action=prev_action,
                state=state,
                mask=mask
            )

            v_loss, clip_frac = self._compute_value_loss(
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=life_mask if self.config.life_mask else None
            )
            va_loss, va_clip_frac = self._compute_value_loss(
                value=value_a, 
                traj_ret=traj_ret_a, 
                old_value=old_value_a, 
                mask=life_mask if self.config.life_mask else None
            )
            n_units = self.model.env_stats.n_units
            loss = self.config.value_coef * (
                1 / (n_units+1) * v_loss + n_units / (n_units+1) * va_loss)

        var = self.encoder.variables[0]
        value_weights = var[:global_state.shape[-1]]
        va_weights = var[global_state.shape[-1]:]

        terms = dict(
            value=value,
            ae_weights=self.action_embed.variables[0], 
            value_weights=value_weights,
            va_weights=va_weights, 
            value_a=value_a,
            v_loss=v_loss,
            va_loss=va_loss,
            value_loss=loss,
            explained_variance=explained_variance(traj_ret, value),
            va_explained_variance=explained_variance(traj_ret_a, value_a),
            traj_ret_std=tf.math.reduce_std(traj_ret, axis=-1), 
            traj_ret_a_std=tf.math.reduce_std(traj_ret_a, axis=-1), 
            v_clip_frac=clip_frac,
            va_clip_frac=va_clip_frac,
        )

        return tape, loss, terms


def create_loss(config, model, name='mappo'):
    def constructor(config, cls, name):
        return cls(
            config=config, 
            model=model[name], 
            name=name)

    return LossEnsemble(
        config=config,
        model=model,
        constructor=constructor,
        name=name,
        policy=MAPPOPolicyLoss,
        value=MAPPOValueLoss,
    )
