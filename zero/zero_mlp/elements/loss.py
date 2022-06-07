import tensorflow as tf

from core.elements.loss import LossEnsemble
from utility.tf_utils import explained_variance, reduce_mean
from algo.gpo.elements.loss import ValueLossImpl, PPOPolicyLoss


class PPOValueLoss(ValueLossImpl):
    def loss(
        self, 
        global_state, 
        action, 
        pi, 
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
        n_units = self.model.env_stats.n_units
        loss_mask = life_mask if self.config.life_mask else None
        n = None if loss_mask is None else tf.reduce_sum(loss_mask)
        with tf.GradientTape() as tape:
            ae, value, value_a, _ = self.model.compute_value(
                global_state=global_state,
                action=action, 
                pi=pi, 
                prev_reward=prev_reward,
                prev_action=prev_action,
                life_mask=life_mask, 
                state=state,
                mask=mask,
                return_ae=True
            )

            raw_v_loss, clip_frac = self._value_loss(
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=loss_mask, 
                reduce=False
            )
            raw_va_loss, va_clip_frac = self._value_loss(
                value=value_a, 
                traj_ret=traj_ret_a, 
                old_value=old_value_a, 
                mask=loss_mask,
                reduce=False,
            )

            v_loss = reduce_mean(raw_v_loss, loss_mask, n)
            va_loss = reduce_mean(raw_va_loss, loss_mask, n)
            if self.config.va_coef == 0: 
                print('Unit-wise value loss')
                value_loss = 1 / (n_units+1) * v_loss \
                    + n_units / (n_units+1) * va_loss
                value_loss = self.config.value_coef * value_loss
            else:
                print('Manual value loss')
                value_loss = self.config.value_coef * v_loss \
                    + self.config.va_coef * va_loss

        if self.model.config.concat_end:
            var = self.value.variables[0]
            value_weights = var[:-ae.shape[-1]]
            va_weights = var[-ae.shape[-1]:]
        else:
            var = self.encoder.variables[0]
            value_weights = var[:global_state.shape[-1]]
            va_weights = var[global_state.shape[-1]:]

        ae_weights = self.action_embed.embedding_vars()
        vev = explained_variance(traj_ret, value)
        vaev = explained_variance(traj_ret_a, value_a)
        terms = dict(
            value=value,
            diff_v=(value - value_a), 
            ae_weights=ae_weights, 
            value_weights=value_weights,
            va_weights=va_weights, 
            value_a=value_a,
            raw_v_loss=raw_v_loss,
            raw_va_loss=raw_va_loss,
            v_loss=v_loss,
            va_loss=va_loss,
            value_loss=value_loss,
            explained_variance=vev,
            va_explained_variance=vaev,
            diff_explained_variance=vev - vaev, 
            traj_ret_per_std=tf.math.reduce_std(traj_ret, axis=-1), 
            traj_ret_a_per_std=tf.math.reduce_std(traj_ret_a, axis=-1), 
            traj_ret_per_max=tf.math.reduce_max(traj_ret, axis=-1), 
            traj_ret_a_per_max=tf.math.reduce_max(traj_ret_a, axis=-1), 
            traj_ret_per_min=tf.math.reduce_min(traj_ret, axis=-1), 
            traj_ret_a_per_min=tf.math.reduce_min(traj_ret_a, axis=-1), 
            value_per_std=tf.math.reduce_std(value, axis=-1), 
            value_per_max=tf.math.reduce_max(value, axis=-1), 
            value_per_min=tf.math.reduce_min(value, axis=-1), 
            value_a_per_std=tf.math.reduce_std(value_a, axis=-1), 
            value_a_per_max=tf.math.reduce_max(value_a, axis=-1), 
            value_a_per_min=tf.math.reduce_min(value_a, axis=-1), 
            v_clip_frac=clip_frac,
            va_clip_frac=va_clip_frac,
            ae=ae, 
        )
        for i in range(self.action_embed.input_dim):
            terms[f'{i}_ae_weights'] = ae_weights[i]

        return tape, value_loss, terms


def create_loss(config, model, name='ppo'):
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
        policy=PPOPolicyLoss,
        value=PPOValueLoss,
    )
