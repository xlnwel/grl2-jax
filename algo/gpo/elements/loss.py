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
    def _ppo_loss(
        self, 
        tape, 
        act_dist,
        action, 
        advantage, 
        logprob, 
        tr_prob,
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
    ):
        with tape.stop_recording():
            is_action_discrete = self.model.policy.is_action_discrete
            raw_adv = advantage
            if self.config.normalize_adv:
                advantage = standard_normalization(
                    advantage, zero_center=self.config.zero_center_adv, 
                    mask=sample_mask, n=n, clip=self.config.adv_clip_range)
            if self.config.process_adv == 'tanh':
                scale = tf.minimum(tf.reduce_max(tf.math.abs(advantage)), 10)
                advantage = scale * tf.math.tanh(advantage * 2 / scale)
            elif self.config.process_adv is None:
                pass
            else:
                raise NotImplementedError(f'Unknown {self.config.process_adv}')
            prob = tf.exp(logprob)

            terms = {}

        self.log_for_debug(
            tape, 
            terms, 
            prob=prob, 
            target_prime_old_diff_prob=target_prob_prime - prob, 
            trust_region_old_diff_prob=tr_prob - prob, 
            trust_region_prime_old_diff_prob=tr_prob_prime - prob, 
            raw_adv=raw_adv, 
            advantage=advantage, 
            raw_adv_unit_std=tf.math.reduce_std(raw_adv, axis=-1), 
            adv_unit_std=tf.math.reduce_std(advantage, axis=-1), 
        )
        if action_mask is not None:
            n_avail_actions = tf.reduce_sum(
                tf.cast(action_mask, tf.float32), -1)
            self.log_for_debug(
                tape, 
                terms, 
                n_avail_actions=n_avail_actions
            )

        tf.debugging.assert_all_finite(advantage, 'Bad advantage')
        new_logprob = act_dist.log_prob(action)
        tf.debugging.assert_all_finite(new_logprob, 'Bad new_logprob')
        raw_entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(raw_entropy, 'Bad entropy')
        log_ratio = new_logprob - logprob
        ratio, loss_pg, loss_clip, raw_ppo_loss, ppo_loss, \
            raw_entropy_loss, entropy_loss, approx_kl, clip_frac = \
            rl_loss.ppo_loss(
                pg_coef=self.config.pg_coef, 
                entropy_coef=self.config.entropy_coef, 
                log_ratio=log_ratio, 
                advantage=advantage, 
                clip_range=self.config.ppo_clip_range, 
                entropy=raw_entropy, 
                mask=sample_mask, 
                n=n, 
            )
        self.log_for_debug(
            tape, 
            terms, 
            ratio=ratio,
            entropy=raw_entropy,
            approx_kl=approx_kl,
            new_logprob=new_logprob, 
            p_clip_frac=clip_frac,
            raw_pg_loss=loss_pg, 
            raw_clipped_loss=loss_clip, 
            raw_ppo_loss=raw_ppo_loss,
            ppo_loss=ppo_loss,
            raw_entropy_loss=raw_entropy_loss, 
            entropy_loss=entropy_loss, 
        )

        # GPO with L1
        new_prob = act_dist.prob(action)
        tr_diff_prob = tr_prob - new_prob
        l1_divident = prob if self.config.get('weighted_l_dist', False) else 1
        l1_distance = tf.math.abs(tr_diff_prob) / l1_divident
        raw_gpo_l1_loss = reduce_mean(l1_distance, mask=sample_mask, n=n)
        tf.debugging.assert_all_finite(raw_gpo_l1_loss, 'Bad raw_gpo_l1_loss')
        gpo_l1_loss = self.config.aux_l1_coef * raw_gpo_l1_loss
        self.log_for_debug(
            tape, 
            terms, 
            new_prob=new_prob, 
            new_old_diff_prob=new_prob - prob, 
            trust_region_new_diff_prob=tr_diff_prob, 
            l1_distance=l1_distance, 
            raw_gpo_l1_loss=raw_gpo_l1_loss, 
            gpo_l1_loss=gpo_l1_loss, 
        )

        # GPO with L2
        l2_divident = prob if self.config.get('weighted_l_dist', False) else 1
        l2_distance = tr_diff_prob**2 / l2_divident
        raw_gpo_l2_loss = reduce_mean(l2_distance, mask=sample_mask, n=n)
        tf.debugging.assert_all_finite(raw_gpo_l2_loss, 'Bad raw_gpo_l2_loss')
        gpo_l2_loss = self.config.aux_l2_coef * raw_gpo_l2_loss
        self.log_for_debug(
            tape, 
            terms, 
            l2_distance=l2_distance, 
            raw_gpo_l2_loss=raw_gpo_l2_loss, 
            gpo_l2_loss=gpo_l2_loss, 
        )

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
                kl_coef=self.config.kl_prior_coef,
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
        self.log_for_debug(
            tape, 
            terms, 
            kl_prior=kl_prior, 
            raw_kl_prior_loss=raw_kl_prior_loss, 
            kl_prior_loss=kl_prior_loss,
        )

        target_prob = locals()[self.config.target_prob]
        target_logprob = tf.math.log(target_prob)
        kl_target, raw_kl_target_loss, kl_target_loss = \
            rl_loss.compute_kl(
                kl_type=self.config.kl_target,
                kl_coef=self.config.kl_target_coef,
                logp=target_logprob,
                logq=new_logprob, 
                sample_prob=prob, 
                pi1=target_pi,
                pi2=new_pi,
                pi_mask=action_mask,
                sample_mask=sample_mask,
                n=n, 
            )
        self.log_for_debug(
            tape, 
            terms, 
            target_prob=target_prob, 
            kl_target=kl_target, 
            raw_kl_target_loss=raw_kl_target_loss, 
            kl_target_loss=kl_target_loss,
        )

        js_target, raw_js_target_loss, js_target_loss = \
            rl_loss.compute_js(
                js_type=self.config.js_target, 
                js_coef=self.config.js_target_coef, 
                p=target_prob, 
                q=new_prob, 
                sample_prob=prob, 
                pi1=target_pi,
                pi2=new_pi,
                pi_mask=action_mask, 
                sample_mask=sample_mask,
                n=n, 
            )
        self.log_for_debug(
            tape, 
            terms, 
            js_target=js_target, 
            raw_js_target_loss=raw_js_target_loss, 
            js_target_loss=js_target_loss,
        )

        tsallis_target, raw_tsallis_target_loss, tsallis_target_loss = \
            rl_loss.compute_tsallis(
                tsallis_type=self.config.tsallis_target,
                tsallis_coef=self.config.tsallis_target_coef,
                tsallis_q=self.config.tsallis_q,
                p=target_prob,
                q=new_prob, 
                sample_prob=prob, 
                pi1=target_pi,
                pi2=new_pi,
                pi_mask=action_mask,
                sample_mask=sample_mask,
                n=n, 
            )
        self.log_for_debug(
            tape, 
            terms, 
            tsallis_target=tsallis_target, 
            raw_tsallis_target_loss=raw_tsallis_target_loss, 
            tsallis_target_loss=tsallis_target_loss, 
        )

        # GPO Loss
        raw_losses = {k: v for k, v in locals().items() 
            if re.search(r'raw_.*_loss$', k)
            and k != 'raw_ppo_loss' and k != 'raw_entropy_loss'}
        losses = {k: v for k, v in locals().items() 
            if re.search(r'^(?!raw_).*_loss$', k) 
            and k != 'ppo_loss' and k != 'entropy_loss'}
        print_dict(raw_losses, prefix='raw losses')
        print_dict(losses, prefix='losses')

        raw_gpo_loss = sum(raw_losses.values())
        gpo_loss = sum(losses.values())
        loss = ppo_loss + entropy_loss + gpo_loss

        self.log_for_debug(
            tape, 
            terms, 
            raw_gpo_loss=raw_gpo_loss, 
            gpo_loss=gpo_loss, 
            loss=loss, 
        )
        terms = prefix_name(terms, name)

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
        loss_mask = life_mask if self.config.life_mask else None
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
            terms, loss = self._ppo_loss(
                tape=tape, 
                act_dist=act_dist, 
                action=action, 
                advantage=advantage, 
                logprob=logprob, 
                tr_prob=tr_prob, 
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

        if self.config.get('debug', True) and life_mask is not None:
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
        loss_mask = life_mask if self.config.life_mask else None
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
