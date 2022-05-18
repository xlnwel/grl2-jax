import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility import rl_loss
from utility.tf_utils import reduce_mean, explained_variance, standard_normalization


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
        vt_prob,
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
        with tape.stop_recording():
            use_gpo_l1 = self.config.get('aux_l1_coef', None) is not None
            use_gpo_l2 = self.config.get('aux_l2_coef', None) is not None
            use_aux_pg = self.config.get('aux_pg_coef', None) is not None
            use_gpo_mix = self.config.get('aux_mix_pg_coef', None) is not None
            is_action_discrete = self.model.policy.is_action_discrete
            raw_adv = advantage
            if self.config.norm_adv:
                advantage = standard_normalization(advantage, mask=mask)
            prob = tf.exp(logprob)

        new_logprob = act_dist.log_prob(action)
        tf.debugging.assert_all_finite(new_logprob, 'Bad new_logprob')
        entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(entropy, 'Bad entropy')
        log_ratio = new_logprob - logprob
        ratio, loss1, loss2, raw_ppo_loss, raw_entropy, approx_kl, clip_frac = \
            rl_loss.ppo_loss(
            log_ratio, 
            advantage, 
            self.config.policy_clip_range, 
            entropy, 
            mask=mask, 
            n=n, 
            reduce=False
        )
        tf.debugging.assert_all_finite(raw_ppo_loss, 'Bad raw_ppo_loss')
        raw_ppo_loss = reduce_mean(raw_ppo_loss, mask, n)
        pg_loss = self.config.pg_coef * raw_ppo_loss
        entropy = reduce_mean(raw_entropy, mask, n)
        entropy_loss = - self.config.entropy_coef * entropy

        # GPO with L1
        new_prob = tf.exp(new_logprob)
        tr_diff_prob = tr_prob - new_prob
        if use_gpo_l1:
            raw_gpo_l1_loss = reduce_mean(tf.math.abs(tr_diff_prob), mask, n)
            tf.debugging.assert_all_finite(raw_gpo_l1_loss, 'Bad raw_gpo_l1_loss')
            gpo_l1_loss = self.config.aux_l1_coef * raw_gpo_l1_loss
        else:
            raw_gpo_l1_loss = 0
            gpo_l1_loss = 0
        
        # GPO with L2
        if use_gpo_l2:
            raw_gpo_l2_loss = reduce_mean(tr_diff_prob**2, mask, n)
            tf.debugging.assert_all_finite(raw_gpo_l2_loss, 'Bad raw_gpo_l2_loss')
            gpo_l2_loss = self.config.aux_l2_coef * raw_gpo_l2_loss
        else:
            raw_gpo_l2_loss = 0
            gpo_l2_loss = 0

        if is_action_discrete:
            new_pi = tf.nn.softmax(act_dist.logits)
            new_pi_mean = None
            new_pi_std = None
        else:
            new_pi = None
            new_pi_mean = act_dist.loc
            new_pi_std = tf.exp(self.model.policy.logstd)

        if use_aux_pg or use_gpo_mix:
            kl = rl_loss.kl_from_probabilities(
                pi1=new_pi, 
                pi2=pi, 
                pi1_mean=new_pi_mean, 
                pi1_std=new_pi_std, 
                pi2_mean=pi_mean,
                pi2_std=pi_std, 
                pi_mask=action_mask
            )
            tf.debugging.assert_all_finite(kl, 'Bad kl')
            tf.debugging.assert_all_finite(approx_kl, 'Bad approx_kl')
            raw_kl_loss = reduce_mean(
                approx_kl if self.config.approx_kl else kl, 
                mask, 
                n
            )
            tf.debugging.assert_all_finite(raw_kl_loss, 'Bad raw_kl_loss')
            kl_prior_loss = self.config.kl_prior_coef * raw_kl_loss
        else:
            raw_kl_loss = 0
            kl_prior_loss = 0

        # GPO with KL
        if use_aux_pg:
            tr_weight = tf.stop_gradient(
                (tr_prob - new_prob) / tf.math.abs(tr_prob - prob))
            vt_weight = tf.stop_gradient(
                (vt_prob - new_prob) / tf.math.abs(vt_prob - prob))
            weighted_aux_adv = raw_adv * vt_weight * tf.sign(raw_adv)
            _, _, _, raw_aux_pg_loss, _, _, _ = rl_loss.ppo_loss(
                log_ratio, 
                weighted_aux_adv, 
                self.config.policy_clip_range, 
                entropy, 
                mask=mask, 
                n=n, 
                reduce=False
            )
            raw_aux_pg_loss = reduce_mean(raw_aux_pg_loss, mask, n)
            tf.debugging.assert_all_finite(raw_aux_pg_loss, 'Bad raw_aux_pg_loss')
            aux_pg_loss = self.config.aux_pg_coef * raw_aux_pg_loss
            gpo_kl_loss = kl_prior_loss + aux_pg_loss
        else:
            raw_aux_pg_loss = 0
            gpo_kl_loss = 0
        
        # GPO with Mix of Forward and Backward KL
        if use_gpo_mix:
            _, _, _, raw_mix_pg_loss, _, _, _ = rl_loss.ppo_loss(
                log_ratio, 
                raw_adv, 
                self.config.policy_clip_range, 
                entropy, 
                mask=mask, 
                n=n, 
                reduce=False
            )
            raw_mix_pg_loss = reduce_mean(raw_mix_pg_loss, mask, n)
            tf.debugging.assert_all_finite(raw_mix_pg_loss, 'Bad raw_mix_pg_loss')
            mix_pg_loss = self.config.up_mix_pg_coef * raw_mix_pg_loss
            gpo_mix_loss = kl_prior_loss + mix_pg_loss
        else:
            raw_mix_pg_loss = 0
            gpo_mix_loss = 0

        # GPO Loss
        if use_gpo_l2 or use_aux_pg:
            raw_gpo_loss = raw_gpo_l1_loss + raw_gpo_l2_loss \
                + raw_kl_loss + raw_aux_pg_loss + raw_mix_pg_loss
            gpo_loss = gpo_l1_loss + gpo_l2_loss + gpo_kl_loss + gpo_mix_loss
            tf.debugging.assert_all_finite(raw_gpo_loss, 'Bad raw_gpo_loss')
        else:
            gpo_loss = 0

        tf.debugging.assert_all_finite(gpo_loss, 'Bad gpo_loss')
        tf.debugging.assert_all_finite(entropy_loss, 'Bad entropy_loss')
        tf.debugging.assert_all_finite(pg_loss, 'Bad pg_loss')
        loss = pg_loss + entropy_loss + gpo_loss

        if self.config.get('debug', True):
            with tape.stop_recording():
                diff_prob = new_prob - prob
                terms = dict(
                    tr_old_diff_prob=tr_prob - prob, 
                    tr_diff_prob=tr_diff_prob, 
                    ratio=tf.exp(log_ratio),
                    raw_entropy=raw_entropy,
                    entropy=entropy,
                    approx_kl=approx_kl,
                    new_logprob=new_logprob, 
                    prob=prob,
                    new_prob=new_prob, 
                    diff_prob=diff_prob, 
                    p_clip_frac=clip_frac,
                    raw_pg_loss=loss1, 
                    raw_clipped_loss=loss2, 
                    raw_ppo_loss=raw_ppo_loss,
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss, 
                    raw_gpo_l2_loss=raw_gpo_l2_loss, 
                    raw_gpo_loss=raw_gpo_loss, 
                    gpo_l2_loss=gpo_l2_loss, 
                    gpo_loss=gpo_loss, 
                    actor_loss=loss,
                    adv_std=tf.math.reduce_std(advantage, axis=-1), 
                )
                if is_action_discrete:
                    tt_prob = tf.gather(target_pi, action, batch_dims=len(action.shape))
                    terms.update(dict(
                        logits=act_dist.logits, 
                        tt_prob=tt_prob, 
                        tt_old_diff_prob=tt_prob - prob, 
                        tr_tt_diff_prob=tr_prob - tt_prob, 
                        tt_diff_prob=tt_prob - new_prob, 
                    ))
                if use_aux_pg or use_gpo_mix:
                    terms.update(dict(
                        kl=kl, 
                        raw_kl_loss=raw_kl_loss, 
                        kl_prior_loss=kl_prior_loss, 
                    ))
                if use_aux_pg:
                    terms.update(dict(
                        tr_weight=tr_weight, 
                        vt_weight=vt_weight, 
                        weighted_aux_adv=weighted_aux_adv, 
                        raw_aux_pg_loss=raw_aux_pg_loss, 
                        vt_old_diff_prob=vt_prob - prob, 
                        aux_pg_loss=aux_pg_loss, 
                        gpo_kl_loss=gpo_kl_loss
                    ))
                if use_gpo_mix:
                    terms.update(dict(
                        raw_mix_pg_loss=raw_mix_pg_loss, 
                        mix_pg_loss=mix_pg_loss, 
                        gpo_mix_loss=gpo_mix_loss
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
                self.config.value_clip_range, 
                mask=mask, 
                n=n,
                reduce=False
            )
        elif value_loss_type == 'clip_huber':
            raw_value_loss, v_clip_frac = rl_loss.clipped_value_loss(
                value, 
                traj_ret, 
                old_value, 
                self.config.value_clip_range, 
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
        vt_prob, 
        target_prob_prime, 
        tr_prob_prime, 
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
                vt_prob=vt_prob, 
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
                t_old_diff_prob=target_prob - terms['prob'], 
                tp_old_diff_prob=target_prob_prime - terms['prob'], 
                trp_old_diff_prob=tr_prob_prime - terms['prob'], 
                tp_t_diff_prob=target_prob_prime - target_prob, 
                trp_tr_diff_prob=tr_prob_prime - tr_prob, 
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
