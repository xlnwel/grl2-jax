import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility import rl_loss
from utility.tf_utils import reduce_mean, explained_variance


def prefix_name(terms, name):
    if name is not None:
        new_terms = {}
        for k, v in terms.items():
            new_terms[f'{name}/{k}'] = v
        return new_terms
    return terms


class PGLossImpl(Loss):
    def _pg_loss(
        self, 
        tape, 
        act_dist,
        action, 
        advantage, 
        tr_prob,
        logprob, 
        target_pi, 
        pi=None, 
        pi_mean=None, 
        pi_std=None, 
        action_mask=None, 
        mask=None, 
        n=None, 
        name=None,
    ):
        use_gpo_l2 = self.config.get('gpo_l2_coef', None) is not None
        use_gpo_kl = self.config.get('gpo_kl_coef', None) is not None
        new_logprob = act_dist.log_prob(action)
        tf.debugging.assert_all_finite(new_logprob, 'Bad new_logprob')
        entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(entropy, 'Bad entropy')
        log_ratio = new_logprob - logprob
        raw_pg_loss, raw_entropy, kl, clip_frac = rl_loss.ppo_loss(
            log_ratio, 
            advantage, 
            self.config.clip_range, 
            entropy, 
            mask=mask, 
            n=n, 
            reduce=False
        )
        tf.debugging.assert_all_finite(raw_pg_loss, 'Bad raw_pg_loss')
        raw_pg_loss = reduce_mean(raw_pg_loss, mask, n)
        pg_loss = self.config.pg_coef * raw_pg_loss
        entropy = reduce_mean(raw_entropy, mask, n)
        entropy_loss = - self.config.entropy_coef * entropy

        # GPO L2
        if use_gpo_l2:
            new_prob = tf.exp(new_logprob)
            tr_diff_prob = tr_prob - new_prob
            raw_gpo_l2_loss = reduce_mean(tr_diff_prob**2, mask, n)
            gpo_l2_loss = self.config.gpo_l2_coef * raw_gpo_l2_loss
        else:
            raw_gpo_l2_loss = 0
            gpo_l2_loss = 0

        # GPO KL
        new_pi = tf.nn.softmax(act_dist.logits)
        if use_gpo_kl:
            gpo_kl = rl_loss.kl_from_probabilities(
                new_pi, target_pi, action_mask)
            raw_gpo_kl_loss = reduce_mean(gpo_kl, mask, n)
            gpo_kl_loss = self.config.gpo_kl_coef * raw_gpo_kl_loss
        else:
            raw_gpo_kl_loss = 0
            gpo_kl_loss = 0

        # GPO Loss
        if use_gpo_l2 or use_gpo_kl:
            raw_gpo_loss = raw_gpo_l2_loss + raw_gpo_kl_loss
            gpo_loss = gpo_l2_loss + gpo_kl_loss
            tf.debugging.assert_all_finite(gpo_loss, 'Bad gpo_loss')
        else:
            gpo_loss = 0

        loss = pg_loss + entropy_loss + gpo_loss

        if self.config.get('debug', True):
            with tape.stop_recording():
                prob = tf.exp(logprob)
                diff_prob = new_prob - prob
                tt_prob = tf.gather(target_pi, action, batch_dims=len(action.shape))
                terms = dict(
                    tr_old_diff_prob=tr_prob - prob, 
                    tt_old_diff_prob=tt_prob - prob, 
                    tr_tt_diff_prob=tr_prob - tt_prob, 
                    tr_diff_prob=tr_diff_prob, 
                    tt_diff_prob=tt_prob - new_prob, 
                    ratio=tf.exp(log_ratio),
                    raw_entropy=raw_entropy,
                    entropy=entropy,
                    kl=kl,
                    new_logprob=new_logprob, 
                    prob=prob,
                    new_prob=new_prob, 
                    diff_prob=diff_prob, 
                    p_clip_frac=clip_frac,
                    raw_pg_loss=raw_pg_loss,
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss, 
                    raw_gpo_l2_loss=raw_gpo_l2_loss, 
                    raw_gpo_loss=raw_gpo_loss, 
                    gpo_l2_loss=gpo_l2_loss, 
                    gpo_loss=gpo_loss, 
                    actor_loss=loss,
                    adv_std=tf.math.reduce_std(advantage, axis=-1), 
                )
                if use_gpo_kl:
                    terms.update(dict(
                        gpo_kl=gpo_kl, 
                    ))
                if action_mask is not None:
                    terms['n_avail_actions'] = tf.reduce_sum(
                        tf.cast(action_mask, tf.float32), -1)
                terms = prefix_name(terms, name)
                if pi is not None:
                    diff_prob = new_pi - pi
                    terms['diff_prob'] = diff_prob
                    max_diff_action = tf.cast(
                        tf.math.argmax(tf.math.abs(diff_prob), axis=-1), tf.int32)
                    terms['diff_match'] = tf.reduce_mean(
                        tf.cast(max_diff_action == action, tf.float32)
                    )
                elif pi_mean is not None:
                    new_mean = act_dist.mean()
                    new_std = tf.exp(self.policy.logstd)
                    terms['new_mean'] = new_mean
                    terms['new_std'] = new_std
                    terms['diff_mean'] = new_mean - pi_mean
                    terms['diff_std'] = new_std - pi_std

        else:
            terms = {}

        return terms, loss


class ValueLossImpl(Loss):
    def _value_loss(
        self, 
        tape, 
        value, 
        traj_ret, 
        old_value, 
        mask=None,
        n=None, 
        name=None, 
    ):
        value_loss_type = getattr(self.config, 'value_loss', 'mse')
        v_clip_frac = 0
        if value_loss_type == 'huber':
            raw_value_loss = rl_loss.huber_loss(
                value, 
                traj_ret, 
                threshold=self.config.huber_threshold
            )
        elif value_loss_type == 'mse':
            raw_value_loss = .5 * (value - traj_ret)**2
        elif value_loss_type == 'clip':
            raw_value_loss, v_clip_frac = rl_loss.clipped_value_loss(
                value, 
                traj_ret, 
                old_value, 
                self.config.clip_range, 
                mask=mask, 
                n=n,
                reduce=False
            )
        elif value_loss_type == 'clip_huber':
            raw_value_loss, v_clip_frac = rl_loss.clipped_value_loss(
                value, 
                traj_ret, 
                old_value, 
                self.config.clip_range, 
                mask=mask, 
                n=n, 
                huber_threshold=self.config.huber_threshold,
                reduce=False
            )
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')

        value_loss = reduce_mean(raw_value_loss, mask)
        loss = reduce_mean(value_loss, mask, n)
        loss = self.config.value_coef * loss

        if self.config.get('debug', True):
            with tape.stop_recording():
                ev = explained_variance(traj_ret, value)
                terms = dict(
                    value=value,
                    raw_v_loss=raw_value_loss,
                    v_loss=loss,
                    explained_variance=ev,
                    traj_ret_std=tf.math.reduce_std(traj_ret, axis=-1), 
                    v_clip_frac=v_clip_frac,
                )
                terms = prefix_name(terms, name)
        else:
            terms = {}

        return terms, loss


class PPOPolicyLoss(PGLossImpl):
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
            terms, loss = self._pg_loss(
                tape=tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                tr_prob=tr_prob,
                logprob=logprob, 
                target_pi=target_pi, 
                pi=pi, 
                pi_mean=pi_mean, 
                pi_std=pi_std, 
                action_mask=action_mask, 
                mask=loss_mask, 
                n=n
            )

        if self.config.get('debug', True):
            terms.update(dict(
                target_old_diff_prob=target_prob - terms['prob'], 
            ))
            if life_mask is not None:
                terms['n_alive_units'] = tf.reduce_sum(
                    life_mask, -1)

        return tape, loss, terms


class PPOValueLoss(ValueLossImpl):
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

            terms, loss = self._value_loss(
                tape=tape, 
                value=value, 
                traj_ret=traj_ret, 
                old_value=old_value, 
                mask=loss_mask,
                n=n,
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
