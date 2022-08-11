import logging
import tensorflow as tf

from core.elements.loss import Loss as LossBase, LossEnsemble
from core.log import do_logging
from utility import rl_loss
from utility.tf_utils import assert_rank_and_shape_compatibility, reduce_mean, \
    explained_variance

logger = logging.getLogger(__name__)

def prefix_name(terms, name):
    if name is not None:
        new_terms = {}
        for k, v in terms.items():
            if '/' not in k:
                new_terms[f'{name}/{k}'] = v
            else:
                new_terms[k] = v
        return new_terms
    return terms


class POLossImpl(LossBase):
    def _pg_loss(
        self, 
        tape, 
        act_dist, 
        advantage, 
        ratio, 
        pi_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        action_mask=None, 
        sample_mask=None, 
        n=None, 
        name=None, 
        use_meta=False, 
        debug=True
    ):
        if not self.config.get('policy_life_mask', True):
            sample_mask = None
        terms = {}
        tf.debugging.assert_all_finite(advantage, 'Bad advantage')
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            advantage=advantage, 
            adv_unit_std=tf.math.reduce_std(advantage, axis=-1), 
        )
        if action_mask is not None:
            n_avail_actions = tf.reduce_sum(
                tf.cast(action_mask, tf.float32), -1)
            self.log_for_debug(
                tape, 
                terms, 
                debug=debug, 
                n_avail_actions=n_avail_actions
            )

        pg_coef = self.model.meta('pg_coef', inner=use_meta)
        entropy_coef = self.model.meta('entropy_coef', inner=use_meta)

        if self.config.use_dice:
            dice_op = rl_loss.dice(
                pi_logprob, 
                axis=self.config.dice_axis, 
                lam=self.config.dice_lam
            )
        else:
            dice_op = pi_logprob
        if self.config.pg_type == 'pg':
            raw_pg_loss, pg_loss = rl_loss.pg_loss(
                pg_coef=pg_coef, 
                advantage=advantage, 
                logprob=dice_op, 
                mask=sample_mask, 
                n=n, 
            )
        elif self.config.pg_type == 'ppo':
            loss_pg, loss_clip, raw_pg_loss, pg_loss, clip_frac = \
                rl_loss.high_order_ppo_loss(
                    pg_coef=pg_coef, 
                    advantage=advantage, 
                    ratio=ratio, 
                    dice_op=dice_op, 
                    clip_range=self.config.ppo_clip_range, 
                    mask=sample_mask, 
                    n=n, 
                )
            self.log_for_debug(
                tape, 
                terms, 
                debug=debug, 
                loss_pg=loss_pg, 
                loss_clip=loss_clip, 
                clip_frac=clip_frac, 
            )
        else:
            raise NotImplementedError
        tf.debugging.assert_all_finite(raw_pg_loss, 'Bad raw_pg_loss')

        entropy = act_dist.entropy()
        tf.debugging.assert_all_finite(entropy, 'Bad entropy')
        raw_entropy_loss, entropy_loss = rl_loss.entropy_loss(
            entropy_coef=entropy_coef, 
            entropy=entropy, 
            mask=sample_mask, 
            n=n
        )
        tf.debugging.assert_all_finite(raw_entropy_loss, 'Bad raw_entropy_loss')
        loss = pg_loss + entropy_loss

        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            dice_op=dice_op, 
            pg_coef=pg_coef, 
            entropy_coef=entropy_coef, 
            entropy=entropy, 
            raw_pg_loss=raw_pg_loss, 
            pg_loss=pg_loss, 
            raw_entropy_loss=raw_entropy_loss, 
            entropy_loss=entropy_loss, 
            po_loss=loss, 
        )
        terms = prefix_name(terms, name)

        return loss, terms


class ValueLossImpl(LossBase):
    def _value_loss(
        self, 
        tape,
        value, 
        target, 
        old_value, 
        sample_mask=None,
        n=None, 
        name=None, 
        use_meta=False, 
        debug=False
    ):
        if not self.config.get('value_life_mask', False):
            sample_mask = None
        value_loss_type = getattr(self.config, 'value_loss', 'mse')
        value_coef = self.model.meta('value_coef', inner=use_meta)
        v_clip_frac = 0
        if value_loss_type == 'huber':
            raw_value_loss = rl_loss.huber_loss(
                value, 
                target, 
                threshold=self.config.huber_threshold
            )
        elif value_loss_type == 'mse':
            raw_value_loss = .5 * (value - target)**2
        elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
            raw_value_loss, v_clip_frac = rl_loss.clipped_value_loss(
                value, 
                target, 
                old_value, 
                self.config.value_clip_range, 
                huber_threshold=self.config.get('huber_threshold', None), 
                mask=sample_mask, 
                n=n,
            )
        else:
            raise ValueError(f'Unknown value loss type: {value_loss_type}')
        raw_value_loss, value_loss = rl_loss.to_loss(
            raw_value_loss, 
            coef=value_coef, 
            mask=sample_mask, 
            n=n
        )

        if debug and self.config.get('debug', True):
            with tape.stop_recording():
                ev = explained_variance(target, value)
                terms = dict(
                    value_coef=value_coef, 
                    value=value,
                    v_target=target, 
                    raw_v_loss=raw_value_loss,
                    v_loss=value_loss,
                    explained_variance=ev,
                    v_target_std=tf.math.reduce_std(target, axis=-1), 
                    v_clip_frac=v_clip_frac, 
                )
                terms = prefix_name(terms, name)
        else:
            terms = {}

        return value_loss, terms


class Loss(ValueLossImpl, POLossImpl):
    def compute_target_advantage(
        self, 
        reward, 
        discount, 
        reset, 
        value, 
        old_value, 
        next_value, 
        ratio, 
        gamma, 
        lam, 
        mask, 
        n
    ):
        if self.config.target_type == 'vtrace':
            v_target, advantage = rl_loss.v_trace_from_ratio(
                reward=reward, 
                value=value, 
                next_value=next_value, 
                ratio=ratio, 
                discount=discount, 
                reset=reset, 
                gamma=gamma, 
                lam=lam, 
                c_clip=self.config.c_clip, 
                rho_clip=self.config.rho_clip, 
                rho_clip_pg=self.config.rho_clip, 
                adv_type=self.config.get('adv_type', 'vtrace'), 
                norm_adv=self.config.get('norm_adv', False), 
                zero_center=self.config.get('zero_center', True), 
                epsilon=self.config.get('epsilon', 1e-8), 
                clip=self.config.get('clip', None), 
                mask=mask, 
                n=n, 
                axis=1, 
            )
        elif self.config.target_type == 'gae':
            v_target, advantage = rl_loss.gae(
                reward=reward, 
                value=value, 
                next_value=next_value, 
                discount=discount, 
                reset=reset, 
                gamma=gamma, 
                lam=lam, 
                norm_adv=self.config.get('norm_adv', False), 
                zero_center=self.config.get('zero_center', True), 
                epsilon=self.config.get('epsilon', 1e-8), 
                clip=self.config.get('clip', None), 
                mask=mask, 
                n=n, 
                axis=1, 
            )
        else:
            raise NotImplementedError
        
        return v_target, advantage

    def loss(
        self, 
        *, 
        tape, 
        obs, 
        idx=None, 
        global_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_global_state=None, 
        action, 
        old_value, 
        reward, 
        discount, 
        reset, 
        mu_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        action_mask=None, 
        sample_mask=None, 
        prev_reward=None,
        prev_action=None,
        state=None, 
        mask=None, 
        name=None, 
        use_meta=None, 
        debug=True, 
    ):
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        gamma = self.model.meta('gamma', inner=use_meta)
        lam = self.model.meta('lam', inner=use_meta)

        x, _ = self.model.encode(
            x=obs, 
            state=state, 
            mask=mask
        )
        if global_state is None:
            global_state = x
        value = self.value(global_state, idx)
        if next_obs is None:
            x = x[:, :-1]
            if idx is not None:
                idx = idx[:, :-1]
            next_value = value[:, 1:]
            value = value[:, :-1]
        else:
            with tape.stop_recording():
                assert state is None, 'unexpected states'
                next_x, _ = self.model.encode(next_obs)
                next_value = self.value(next_x, next_idx)
        act_dist = self.policy(x, hx=idx, action_mask=action_mask)
        pi_logprob = act_dist.log_prob(action)
        assert_rank_and_shape_compatibility([pi_logprob, mu_logprob])
        log_ratio = pi_logprob - mu_logprob
        ratio = tf.exp(log_ratio)
        # tf.debugging.assert_near(
        #     tf.where(tf.cast(reset, bool), 0., log_ratio), 0., 1e-5, 1e-5)

        with tape.stop_recording():
            v_target, advantage = self.compute_target_advantage(
                reward=reward, 
                discount=discount, 
                reset=reset, 
                value=value, 
                old_value=old_value, 
                next_value=next_value, 
                ratio=ratio, 
                gamma=gamma, 
                lam=lam, 
                mask=sample_mask, 
                n=n
            )

        actor_loss, actor_terms = self._pg_loss(
            tape=tape, 
            act_dist=act_dist, 
            advantage=advantage, 
            ratio=ratio, 
            pi_logprob=pi_logprob, 
            mu=mu, 
            mu_mean=mu_mean, 
            mu_std=mu_std, 
            action_mask=action_mask, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )
        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value,
            target=v_target, 
            old_value=old_value, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        loss = actor_loss + value_loss
        
        terms = {**actor_terms, **value_terms}
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma, 
            lam=lam, 
            pi_logprob=pi_logprob, 
            ratio=ratio, 
            approx_kl=.5 * reduce_mean((log_ratio)**2, sample_mask, n), 
            loss=loss, 
        )

        if debug:
            with tape.stop_recording():
                if sample_mask is not None:
                    n_alive_units = tf.reduce_sum(sample_mask, -1)
                    terms['n_alive_units'] = n_alive_units
                if mu is not None:
                    terms['diff_pi'] = tf.nn.softmax(act_dist.logits) - mu
                elif mu_mean is not None:
                    terms['pi_mean'] = act_dist.loc
                    terms['diff_pi_mean'] = act_dist.loc - mu_mean
                    pi_std = tf.exp(self.policy.logstd)
                    terms['pi_std'] = pi_std
                    terms['diff_pi_std'] = pi_std - mu_std
                    # tf.debugging.assert_equal(pi_std, mu_std)
                    # tf.debugging.assert_equal(act_dist.loc, mu_mean)
        terms = prefix_name(terms, name)

        return loss, terms

    def bmg_loss(
        self, 
        *, 
        tape, 
        obs, 
        idx=None, 
        hidden_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_hidden_state=None, 
        action, 
        old_value, 
        reward, 
        discount, 
        reset, 
        mu_logprob, 
        mu=None, 
        mu_mean=None, 
        mu_std=None, 
        state=None, 
        action_mask=None, 
        sample_mask=None, 
        n=None, 
        mask=None, 
        name=None, 
        use_meta=None, 
        debug=True, 
    ):
        _, act_dist, value = self.model.forward(
            obs=obs[:, :-1] if next_obs is None else obs, 
            idx=idx[:, :-1] if next_idx is None else idx, 
            global_state=hidden_state[:, :-1] if hidden_state is None else hidden_state, 
            state=state, 
            mask=mask
        )
        logprob = act_dist.prob(action)
        sample_prob = tf.exp(mu_logprob)

        is_action_discrete = self.model.policy.is_action_discrete
        if is_action_discrete:
            pi = tf.nn.softmax(act_dist.logits)
            pi_mean = None
            pi_std = None
        else:
            pi = None
            pi_mean = act_dist.loc
            pi_std = tf.exp(self.model.policy.logstd)

        kl, raw_kl_loss, kl_loss = rl_loss.compute_kl(
            kl_type=self.config.kl,
            kl_coef=self.config.kl_coef,
            logp=mu_logprob, 
            logq=logprob, 
            sample_prob=sample_prob, 
            pi1=mu,
            pi2=pi,
            pi1_mean=mu_mean,
            pi2_mean=pi_mean,
            pi1_std=mu_std,
            pi2_std=pi_std,
            sample_mask=sample_mask,
            n=n, 
            pi_mask=action_mask,
        )
        actor_loss = kl_loss

        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value, 
            target=old_value, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        loss = actor_loss + value_loss

        terms = value_terms
        self.log_for_debug(
            tape, 
            terms, 
            kl=kl, 
            raw_kl_loss=raw_kl_loss, 
            kl_loss=kl_loss, 
            actor_loss=actor_loss, 
            loss=loss, 
        )
        terms = prefix_name(terms, name)

        return loss, terms

def create_loss(config, model, name='zero'):
    rl_loss = Loss(config=config.rl, model=model['rl'], name='rl')
    meta_loss = Loss(config=config.meta, model=model['meta'], name='meta')

    return LossEnsemble(
        config=config, 
        model=model, 
        components=dict(
            rl=rl_loss, 
            meta=meta_loss
        ), 
        name=name, 
    )
