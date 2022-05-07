import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility.rl_loss import huber_loss, reduce_mean, ppo_loss, clipped_value_loss
from utility.tf_utils import explained_variance


class PPOLossImpl(Loss):
    def _compute_value_loss(
        self, 
        value, 
        traj_ret, 
        old_value, 
        mask=None,
        reduce=True,
    ):
        value_loss_type = getattr(self.config, 'value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'huber':
            value_loss = huber_loss(
                value, traj_ret, threshold=self.config.huber_threshold)
        elif value_loss_type == 'mse':
            value_loss = .5 * (value - traj_ret)**2
        elif value_loss_type == 'clip':
            value_loss, v_clip_frac = clipped_value_loss(
                value, traj_ret, old_value, self.config.clip_range, 
                mask=mask, reduce=False)
        elif value_loss_type == 'clip_huber':
            value_loss, v_clip_frac = clipped_value_loss(
                value, traj_ret, old_value, self.config.clip_range, 
                mask=mask, huber_threshold=self.config.huber_threshold,
                reduce=False)
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')

        if reduce:
            value_loss = reduce_mean(value_loss, mask)
        return value_loss, v_clip_frac


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
        target_pi, 
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
            tf.debugging.assert_all_finite(new_logprob, 'Bad new_logprob')
            entropy = act_dist.entropy()
            tf.debugging.assert_all_finite(entropy, 'Bad entropy')
            log_ratio = new_logprob - logprob
            raw_pg_loss, raw_entropy, kl, clip_frac = ppo_loss(
                log_ratio, 
                advantage, 
                self.config.clip_range, 
                entropy, 
                mask=loss_mask, 
                n=n, 
                reduce=False
            )
            tf.debugging.assert_all_finite(raw_pg_loss, 'Bad raw_pg_loss')
            raw_pg_loss = reduce_mean(raw_pg_loss, loss_mask, n)
            pg_loss = self.config.pg_coef * raw_pg_loss
            entropy = reduce_mean(raw_entropy, loss_mask, n)
            entropy_loss = - self.config.entropy_coef * entropy

            # GPO L2
            new_prob = tf.exp(new_logprob)
            tr_diff_prob = tr_prob - new_prob
            tf.debugging.assert_all_finite(act_dist.logits, 'Bad logits')
            new_pi = tf.nn.softmax(act_dist.logits)
            tf.debugging.assert_all_finite(new_pi, 'Bad new pi')
            raw_gpo_l2_loss = reduce_mean(tr_diff_prob**2, loss_mask, n)
            gpo_l2_loss = self.config.gpo_l2_coef * raw_gpo_l2_loss

            # GPO KL
            ratio = new_pi / target_pi
            if action_mask is not None:
                ratio = tf.where(action_mask, ratio, 1)
            tf.debugging.assert_all_finite(ratio, 'Bad ratio')
            log_ratio = tf.math.log(ratio)
            tf.debugging.assert_all_finite(log_ratio, 'Bade log_ratio')
            gpo_kl = tf.reduce_sum(
                tf.math.multiply_no_nan(new_pi, log_ratio), axis=-1)
            tf.debugging.assert_all_finite(gpo_kl, 'Bad gpo_kl')
            raw_gpo_kl_loss = reduce_mean(gpo_kl, loss_mask, n)
            gpo_kl_loss = self.config.gpo_kl_coef * raw_gpo_kl_loss
            raw_gpo_loss = raw_gpo_l2_loss + raw_gpo_kl_loss
            gpo_loss = gpo_l2_loss + gpo_kl_loss
            tf.debugging.assert_all_finite(gpo_loss, 'Bad gpo_loss')

            loss = pg_loss + entropy_loss + gpo_l2_loss

        prob = tf.exp(logprob)
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
            gpo_kl=gpo_kl, 
            logprob=logprob,
            new_logprob=new_logprob, 
            prob=prob,
            new_prob=new_prob, 
            diff_prob=diff_prob, 
            p_clip_frac=clip_frac,
            raw_pg_loss=raw_pg_loss,
            pg_loss=pg_loss,
            entropy_loss=entropy_loss, 
            raw_gpo_l2_loss=raw_gpo_l2_loss, 
            raw_gpo_kl_loss=raw_gpo_kl_loss, 
            raw_gpo_loss=raw_gpo_loss, 
            gpo_l2_loss=gpo_l2_loss, 
            gpo_kl_loss=gpo_kl_loss, 
            gpo_loss=gpo_loss, 
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
            max_diff_prob = tf.math.reduce_max(new_pi - pi, axis=-1)
            terms['new_pi'] = new_pi
            terms['max_diff_prob'] = max_diff_prob
            terms['diff_match'] = tf.reduce_mean(
                tf.cast(tf.math.abs(diff_prob - max_diff_prob) < .01, 
                tf.float32)
            )
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
        value, 
        traj_ret, 
        prev_reward=None, 
        prev_action=None, 
        state=None, 
        life_mask=None, 
        mask=None
    ):
        old_value = value
        loss_mask = life_mask if self.config.value_life_mask else None
        n = None if loss_mask is None else tf.reduce_sum(loss_mask)
        with tf.GradientTape() as tape:
            value, _ = self.model.compute_value(
                global_state=global_state,
                prev_reward=prev_reward,
                prev_action=prev_action,
                state=state,
                mask=mask
            )

            raw_loss, clip_frac = self._compute_value_loss(
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=loss_mask,
                reduce=False
            )
            loss = reduce_mean(raw_loss, loss_mask, n)
            loss = self.config.value_coef * loss

        ev = explained_variance(traj_ret, value)
        terms = dict(
            value=value,
            raw_v_loss=raw_loss,
            v_loss=loss,
            explained_variance=ev,
            traj_ret_std=tf.math.reduce_std(traj_ret, axis=-1), 
            v_clip_frac=clip_frac,
        )

        return tape, loss, terms


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
