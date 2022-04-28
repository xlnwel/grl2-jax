import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility.rl_loss import ppo_loss
from utility.tf_utils import explained_variance, reduce_mean
from algo.hm.elements.loss import PPOLossImpl


class PPOPolicyLoss(Loss):
    def loss(
        self, 
        obs, 
        action, 
        advantage, 
        target_prob, 
        tr_prob, 
        logprob, 
        pi, 
        pi_mean, 
        pi_std, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        action_mask=None, 
        life_mask=None, 
        mask=None
    ):
        loss_mask = life_mask if self.config.policy_life_mask else None
        n = None if loss_mask is None else tf.reduce_sum(loss_mask)
        with tf.GradientTape() as tape:
            x, _ = self.model.encode(
                x=obs, 
                prev_reward=prev_reward, 
                prev_action=prev_action, 
                state=state, 
                mask=mask
            )
            act_dist = self.policy(x, action_mask)
            new_logprob = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logprob - logprob
            raw_policy_loss, raw_entropy, kl, clip_frac = ppo_loss(
                log_ratio, 
                advantage, 
                self.config.clip_range, 
                entropy, 
                mask=loss_mask, 
                n=n, 
                reduce=False
            )
            policy_loss = reduce_mean(raw_policy_loss, loss_mask, n)
            entropy = reduce_mean(raw_entropy, loss_mask, n)
            entropy_loss = - self.config.entropy_coef * entropy
            new_prob = tf.exp(new_logprob)
            tr_diff_prob = new_prob - tr_prob
            raw_maca_loss = reduce_mean(tr_diff_prob**2, mask)
            maca_loss = self.config.maca_coef * raw_maca_loss
            loss = policy_loss + entropy_loss + maca_loss
        
        prob = tf.exp(logprob)
        new_prob = tf.exp(new_logprob)
        diff_prob = new_prob - prob
        terms = dict(
            target_old_diff_prob=target_prob - prob, 
            tr_old_diff_prob=tr_prob - prob, 
            tr_diff_prob=tr_diff_prob, 
            prev_reward=prev_reward,
            ratio=tf.exp(log_ratio),
            raw_entropy=raw_entropy,
            entropy=entropy,
            kl=kl,
            logprob=logprob,
            new_logprob=new_logprob, 
            prob=prob,
            new_prob=new_prob, 
            diff_prob=diff_prob, 
            p_clip_frac=clip_frac,
            raw_policy_loss=raw_policy_loss,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss, 
            raw_maca_loss=raw_maca_loss, 
            maca_loss=maca_loss, 
            actor_loss=loss,
            adv_std=tf.math.reduce_std(advantage, axis=-1), 
        )
        if action_mask is not None:
            terms['n_avail_actions'] = tf.reduce_sum(
                tf.cast(action_mask, tf.float32), -1)
        if life_mask is not None:
            terms['n_alive_units'] = tf.reduce_sum(
                life_mask, -1)
        if self.policy.is_action_discrete:
            new_pi = tf.nn.softmax(act_dist.logits)
            max_diff_prob = tf.math.reduce_max(new_pi - pi, axis=-1)
            terms['new_pi'] = new_pi
            terms['max_diff_prob'] = max_diff_prob
            terms['diff_match'] = tf.math.equal(diff_prob, max_diff_prob)
        else:
            new_mean = act_dist.mean()
            new_std = tf.exp(self.policy.logstd)
            terms['new_mean'] = new_mean
            terms['new_std'] = new_std
            terms['diff_mean'] = new_mean - pi_mean
            terms['diff_std'] = new_std - pi_std

        return tape, loss, terms


class PPOValueLoss(PPOLossImpl):
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
        loss_mask = life_mask if self.config.value_life_mask else None
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

            raw_v_loss, clip_frac = self._compute_value_loss(
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=loss_mask, 
                reduce=False
            )
            raw_va_loss, va_clip_frac = self._compute_value_loss(
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
