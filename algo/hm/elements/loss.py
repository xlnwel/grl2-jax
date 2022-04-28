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


class MAPPOPolicyLoss(Loss):
    def loss(
        self, 
        obs, 
        action, 
        advantage, 
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
            loss = policy_loss + entropy_loss

        prob = tf.exp(logprob)
        new_prob = tf.exp(new_logprob)
        diff_prob = new_prob - prob
        terms = dict(
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


class MAPPOValueLoss(PPOLossImpl):
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
