import re
import tensorflow as tf

from core.elements.loss import Loss, LossEnsemble
from utility import rl_loss
from utility.display import print_dict
from utility.tf_utils import reduce_mean, explained_variance, standard_normalization


def prefix_name(terms, name):
    if name is not None:
        new_terms = {}
        for k, v in terms.items():
            new_terms[f'{name}/{k}'] = v
        return new_terms
    return terms


class GPOLossImpl(Loss):
    def _pg_loss(
        self, 
        tape, 
        act_dist,
        action, 
        advantage, 
        logprob, 
        tr_prob,
        vt_prob,
        target_prob_prime, 
        tr_prob_prime, 
        target_pi, 
        pi=None, 
        pi_mean=None, 
        pi_std=None, 
        action_mask=None, 
        sample_mask=None, 
        n=None, 
        name=None,
        use_meta=False
    ):
        with tape.stop_recording():
            use_gpo_l1 = self.config.get('aux_l1_coef', None) is not None
            use_gpo_l2 = self.config.get('aux_l2_coef', None) is not None
            use_new_po = self.config.get('new_po_coef', None) is not None
            use_kl_prior = self.config.get('kl_prior_coef', None) is not None
            use_kl_target = self.config.get('kl_target_coef', None) is not None
            use_js = self.config.get('js_coef', None) is not None
            is_action_discrete = self.model.policy.is_action_discrete
            raw_adv = advantage
            if self.config.normalize_adv:
                advantage = standard_normalization(
                    advantage, mask=sample_mask, n=n, clip=self.config.clip)
            if self.config.process_adv == 'tanh':
                scale = tf.minimum(tf.reduce_max(tf.math.abs(advantage)), 10)
                advantage = scale * tf.math.tanh(advantage * 2 / scale)
            elif self.config.process_adv is None:
                pass
            else:
                raise NotImplementedError(f'Unknown {self.config.process_adv}')
            prob = tf.exp(logprob)

        pg_coef = self.model.meta('pg_coef', trainable=use_meta)
        kl_prior_coef = self.model.meta('kl_prior_coef', trainable=use_meta)
        kl_target_coef = self.model.meta('kl_target_coef', trainable=use_meta)
        js_target_coef = self.model.meta('js_target_coef', trainable=use_meta)


        tf.debugging.assert_all_finite(advantage, 'Bad advantage')
        new_logprob = act_dist.log_prob(action)
        tf.debugging.assert_all_finite(new_logprob, 'Bad new_logprob')
        entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(entropy, 'Bad entropy')
        log_ratio = new_logprob - logprob
        ratio, loss1, loss2, raw_pg, entropy, approx_kl, clip_frac = \
            rl_loss.ppo_loss(
            log_ratio, 
            advantage, 
            self.config.ppo_clip_range, 
            entropy, 
            mask=sample_mask, 
            n=n, 
            reduce=False
        )
        tf.debugging.assert_all_finite(raw_pg, 'Bad raw_ppo_loss')
        raw_pg_loss = reduce_mean(raw_pg, sample_mask, n)
        pg_loss = pg_coef * reduce_mean(raw_pg, sample_mask, n)
        raw_entropy_loss = - reduce_mean(entropy, sample_mask, n)
        entropy_loss = self.config.entropy_coef * raw_entropy_loss

        # GPO with L1
        new_prob = act_dist.prob(action)
        tr_diff_prob = tr_prob - new_prob
        vt_diff_prob = vt_prob - new_prob
        if use_gpo_l1:
            if self.config.get('weighted_l_dist', False):
                l1_distance = tf.math.abs(tr_diff_prob) / prob
            else:
                l1_distance = tf.math.abs(tr_diff_prob)
            raw_gpo_l1_loss = reduce_mean(
                tf.math.abs(l1_distance), sample_mask, n)
            tf.debugging.assert_all_finite(raw_gpo_l1_loss, 'Bad raw_gpo_l1_loss')
            gpo_l1_loss = self.config.aux_l1_coef * raw_gpo_l1_loss
        else:
            raw_gpo_l1_loss = 0.
            gpo_l1_loss = 0.

        # GPO with L2
        if use_gpo_l2:
            if self.config.get('weighted_l_dist', False):
                l2_distance = tr_diff_prob**2 / prob
            else:
                l2_distance = tr_diff_prob**2
            raw_gpo_l2_loss = reduce_mean(l2_distance, sample_mask, n)
            tf.debugging.assert_all_finite(raw_gpo_l2_loss, 'Bad raw_gpo_l2_loss')
            gpo_l2_loss = self.config.aux_l2_coef * raw_gpo_l2_loss
        else:
            raw_gpo_l2_loss = 0.
            gpo_l2_loss = 0.

        if is_action_discrete:
            new_pi = tf.nn.softmax(act_dist.logits)
            new_pi_mean = None
            new_pi_std = None
        else:
            new_pi = None
            new_pi_mean = act_dist.loc
            new_pi_std = tf.exp(self.model.policy.logstd)

        kl_prior, raw_kl_prior_loss, kl_prior_loss = \
            rl_loss.compute_kl(
                kl_type=self.config.kl_prior,
                kl_coef=kl_prior_coef,
                logp=logprob,
                logq=new_logprob, 
                sample_prob=prob, 
                pi1=pi,
                pi2=new_pi,
                pi1_mean=pi_mean,
                pi2_mean=new_pi_mean,
                pi1_std=pi_std,
                pi2_std=new_pi_std,
                sample_mask=sample_mask,
                n=n, 
                pi_mask=action_mask,
            )

        target_prob = locals()[self.config.target_prob]
        target_logprob = tf.math.log(target_prob)
        kl_target, raw_kl_target_loss, kl_target_loss = \
            rl_loss.compute_kl(
                kl_type=self.config.kl_target,
                kl_coef=kl_target_coef,
                logp=target_logprob,
                logq=new_logprob, 
                sample_prob=prob, 
                pi1=target_pi,
                pi2=new_pi,
                pi_mask=action_mask,
                sample_mask=sample_mask,
                n=n, 
            )

        js, raw_js_loss, js_loss = rl_loss.compute_js(
            js_type=self.config.js_target, 
            js_coef=js_target_coef, 
            logp=new_logprob, 
            logq=target_logprob, 
            sample_prob=prob, 
            pi_mask=action_mask, 
            sample_mask=sample_mask,
            n=n, 
        )

        # GPO's loss
        vt_old_diff_prob = vt_prob - prob
        if use_new_po:
            in_range = tf.sign(vt_diff_prob) == tf.sign(vt_old_diff_prob)
            weighted_l1_po = advantage / prob * tf.where(
                in_range, tf.abs(vt_diff_prob), 0)
            raw_new_po_loss = reduce_mean(weighted_l1_po, sample_mask, n)
            tf.debugging.assert_all_finite(raw_new_po_loss, 'Bad raw_new_po_loss')
            new_po_loss = self.config.new_po_coef * raw_new_po_loss
        else:
            raw_new_po_loss = 0.
            new_po_loss = 0.

        # GPO Loss
        raw_losses = {k: v for k, v in locals().items() 
            if re.search(r'raw_.*_loss$', k)
            and k != 'raw_pg_loss' and k != 'raw_entropy_loss'}
        losses = {k: v for k, v in locals().items() 
            if re.search(r'^(?!raw_).*_loss$', k) 
            and k != 'pg_loss' and k != 'entropy_loss'}
        print_dict(raw_losses, prefix='raw losses')
        print_dict(losses, prefix='losses')

        raw_gpo_loss = sum(raw_losses.values())
        gpo_loss = sum(losses.values())
        loss = pg_loss + entropy_loss + gpo_loss

        if self.config.get('debug', True):
            with tape.stop_recording():
                diff_prob = new_prob - prob
                terms = dict(
                    new_old_diff_prob=diff_prob, 
                    trust_region_old_diff_prob=tr_prob - prob, 
                    trust_region_new_diff_prob=tr_diff_prob, 
                    ratio=ratio,
                    entropy=entropy,
                    approx_kl=approx_kl,
                    prob=prob,
                    new_logprob=new_logprob, 
                    new_prob=new_prob, 
                    p_clip_frac=clip_frac,
                    ppo_loss1=loss1, 
                    ppo_loss2=loss2, 
                    raw_clipped_loss=loss2, 
                    raw_pg_loss=raw_pg_loss,
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss, 
                    raw_gpo_loss=raw_gpo_loss, 
                    gpo_loss=gpo_loss, 
                    actor_loss=loss,
                    advantage=advantage, 
                    raw_adv_std=tf.math.reduce_std(raw_adv, axis=-1), 
                    adv_std=tf.math.reduce_std(advantage, axis=-1), 
                )
                if is_action_discrete:
                    tt_prob = tf.gather(target_pi, action, batch_dims=len(action.shape))
                    tf.debugging.assert_equal(tt_prob, vt_prob)
                    terms.update(dict(
                        logits=act_dist.logits, 
                        valid_target_prob=vt_prob, 
                        valid_target_old_diff_prob=vt_old_diff_prob, 
                        valid_target_new_diff_prob=vt_diff_prob, 
                        valid_target_old_prob_ratio=vt_prob / prob, 
                        valid_target_new_prob_ratio=vt_prob / new_prob, 
                        valid_target_new_pi_ratio=target_pi / new_pi, 
                    ))
                    if pi is not None:
                        diff_pi = new_pi - pi
                        terms['diff_pi'] = diff_pi
                        max_diff_action = tf.cast(
                            tf.math.argmax(tf.math.abs(diff_pi), axis=-1), tf.int32)
                        terms['diff_match'] = tf.reduce_mean(
                            tf.cast(max_diff_action == action, tf.float32)
                        )
                elif pi_mean is not None:
                    new_mean = act_dist.mean()
                    new_std = tf.exp(self.policy.logstd)
                    terms['new_pi_mean'] = new_mean
                    terms['new_pi_std'] = new_std
                    terms['diff_pi_mean'] = new_mean - pi_mean
                    terms['diff_pi_std'] = new_std - pi_std

                if use_gpo_l1:
                    terms.update(dict(
                        l1_distance=l1_distance, 
                        raw_gpo_l1_loss=raw_gpo_l1_loss, 
                        gpo_l1_loss=gpo_l1_loss, 
                    ))
                if use_gpo_l2:
                    terms.update(dict(
                        l2_distance=l2_distance, 
                        raw_gpo_l2_loss=raw_gpo_l2_loss, 
                        gpo_l2_loss=gpo_l2_loss, 
                    ))
                if use_kl_prior:
                    terms.update(dict(
                        kl_prior=kl_prior, 
                        raw_kl_prior_loss=raw_kl_prior_loss, 
                        kl_prior_loss=kl_prior_loss, 
                    ))
                if use_kl_target:
                    terms.update(dict(
                        kl_target=kl_target, 
                        raw_kl_target_loss=raw_kl_target_loss, 
                        kl_target_loss=kl_target_loss, 
                    ))
                if use_new_po:
                    terms.update(dict(
                        out_of_region_frac=1 - tf.cast(in_range, tf.float32), 
                        weighted_l1_po=weighted_l1_po, 
                        raw_new_po_loss=raw_new_po_loss, 
                        new_po_loss=new_po_loss
                    ))
                if use_js:
                    terms.update(dict(
                        js=js, 
                        raw_js_loss=raw_js_loss,
                        js_loss=js_loss,
                    ))
                if action_mask is not None:
                    terms['n_avail_actions'] = tf.reduce_sum(
                        tf.cast(action_mask, tf.float32), -1)
                terms = prefix_name(terms, name)
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
        sample_mask=None,
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
                mask=sample_mask, 
                n=n,
                reduce=False
            )
        elif value_loss_type == 'clip_huber':
            raw_value_loss, v_clip_frac = rl_loss.clipped_value_loss(
                value, 
                traj_ret, 
                old_value, 
                self.config.value_clip_range, 
                mask=sample_mask, 
                n=n, 
                huber_threshold=self.config.huber_threshold,
                reduce=False
            )
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')

        loss = reduce_mean(raw_value_loss, sample_mask, n)
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


class GPOPolicyLoss(GPOLossImpl):
    def loss(
        self, 
        obs, 
        action, 
        advantage, 
        logprob, 
        target_prob, 
        tr_prob, 
        vt_prob, 
        target_prob_prime, 
        tr_prob_prime, 
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
                logprob=logprob, 
                tr_prob=tr_prob, 
                vt_prob=vt_prob, 
                target_prob_prime=target_prob_prime, 
                tr_prob_prime=tr_prob_prime, 
                pi=pi, 
                target_pi=target_pi, 
                pi_mean=pi_mean, 
                pi_std=pi_std, 
                action_mask=action_mask, 
                sample_mask=loss_mask, 
                n=n
            )

        if self.config.get('debug', True):
            terms.update(dict(
                target_prime_old_diff_prob=target_prob_prime - terms['prob'], 
                trust_prime_old_diff_prob=tr_prob_prime - terms['prob'], 
                target_prime_target_diff_prob=target_prob_prime - target_prob, 
                trust_prime_trust_diff_prob=tr_prob_prime - tr_prob, 
                target_prime_trust_prime_diff_prob=target_prob_prime - tr_prob_prime, 
                target_trust_diff_prob=target_prob - tr_prob, 
                target_prime_valid_target_diff_prob=target_prob_prime - vt_prob, 
                trust_prime_valid_target_diff_prob=tr_prob_prime - vt_prob, 
                target_valid_target_diff_prob=target_prob - vt_prob, 
                trust_valid_target_diff_prob=tr_prob - vt_prob, 
            ))
            if life_mask is not None:
                terms['n_alive_units'] = tf.reduce_sum(
                    life_mask, -1)

        return tape, loss, terms


class GPOValueLoss(ValueLossImpl):
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
                sample_mask=loss_mask,
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
        policy=GPOPolicyLoss,
        value=GPOValueLoss,
    )
