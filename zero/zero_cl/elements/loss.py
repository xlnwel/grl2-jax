import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility.rl_loss import ppo_loss, reduce_mean
from utility.tf_utils import explained_variance
from algo.gpo.elements.loss import ValueLossImpl


class MAPPOPolicyLoss(Loss):
    def loss(
        self, 
        obs, 
        action, 
        advantage, 
        logpi, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            act_dist = self.policy(x, action_mask)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, clip_frac = ppo_loss(
                log_ratio, advantage, self.config.clip_range, entropy, 
                life_mask if self.config.life_mask else None)
            entropy_loss = - self.config.entropy_coef * entropy
            loss = policy_loss + entropy_loss

        pi = tf.exp(logpi)
        new_pi = tf.exp(new_logpi)
        terms = dict(
            prev_reward=prev_reward,
            ratio=tf.exp(log_ratio),
            entropy=entropy,
            kl=kl,
            logpi=logpi,
            new_logpi=new_logpi, 
            pi=pi,
            new_pi=new_pi, 
            diff_pi=new_pi - pi, 
            p_clip_frac=clip_frac,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss, 
            actor_loss=loss,
            adv_std=tf.math.reduce_std(advantage, axis=-1), 
        )
        if action_mask is not None:
            terms['n_avail_actions'] = tf.reduce_sum(tf.cast(action_mask, tf.float32), -1)
        if life_mask is not None:
            terms['life_mask'] = life_mask
        if not self.policy.is_action_discrete:
            terms['std'] = tf.exp(self.policy.logstd)

        return tape, loss, terms


class MAPPOValueLoss(ValueLossImpl):
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
        n_units = self.model.env_stats.n_units
        with tf.GradientTape() as tape:
            ae, value, value_a, _ = self.model.compute_value(
                global_state=global_state,
                action=action, 
                prev_reward=prev_reward,
                prev_action=prev_action,
                state=state,
                mask=mask,
                return_ae=True
            )

            v_loss, clip_frac = self._value_loss(
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=life_mask if self.config.life_mask else None
            )
            va_loss, va_clip_frac = self._value_loss(
                value=value_a, 
                traj_ret=traj_ret_a, 
                old_value=old_value_a, 
                mask=life_mask if self.config.life_mask else None
            )

            value_loss = self.config.value_coef * (
                1 / (n_units+1) * v_loss 
                + n_units / (n_units+1) * va_loss
            )
            # value_loss = self.config.value_coef * v_loss \
            #     + self.config.va_coef * va_loss

        var = self.encoder.variables[0]
        value_weights = var[:global_state.shape[-1]]
        va_weights = var[global_state.shape[-1]:]
        # var = self.value.variables[0]
        # value_weights = var[:256]
        # va_weights = var[256:]

        ae_weights = self.action_embed.embedding_vars()
        terms = dict(
            value=value,
            v_diff=(value - value_a), 
            ae_weights=ae_weights, 
            value_weights=value_weights,
            va_weights=va_weights, 
            value_a=value_a,
            v_loss=v_loss,
            va_loss=va_loss,
            value_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            va_explained_variance=explained_variance(traj_ret_a, value_a),
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

    def negative_loss(
        self, 
        global_state, 
        neg_action, 
        traj_ret_a, 
        ae, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        life_mask=None, 
        mask=None
    ):
        with tf.GradientTape() as tape:
            neg_ae = self.model.compute_action_embedding(global_state, neg_action)
            value_na, _ = self.model.compute_value_with_action_embeddings(
                global_state=global_state, 
                ae=neg_ae,
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )

            vna_target = tf.math.minimum(
                traj_ret_a * (1 - self.config.vna_clip_range), 
                traj_ret_a * (1 + self.config.vna_clip_range), 
            )
            diff_vna = tf.maximum(value_na - vna_target, 0)
            vna_loss = .5 * self.config.vna_coef * reduce_mean(
                diff_vna**2, 
                mask=life_mask if self.config.life_mask else None
            )
            clip_frac = reduce_mean(
                tf.cast(tf.math.less_equal(value_na, vna_target), diff_vna.dtype), mask)

        diff_ae = ae - neg_ae
        terms = dict(
            diff_ae=diff_ae, 
            diff_ae_norm=tf.norm(diff_ae, axis=-1), 
            diff_vna=diff_vna, 
            vna_target=vna_target, 
            vna_clip_frac=clip_frac, 
            value_na=value_na, 
            vna_loss=vna_loss
        )

        return tape, vna_loss, terms


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
