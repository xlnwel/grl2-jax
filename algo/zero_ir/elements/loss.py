import logging
import tensorflow as tf

from core.elements.loss import Loss as LossBase, LossEnsemble
from core.log import do_logging
from utility import rl_loss
from utility.tf_utils import assert_rank_and_shape_compatibility, reduce_mean, \
    explained_variance
from .utils import get_hx

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
        next_value, 
        ratio, 
        gamma, 
        lam, 
        norm_adv, 
        mask=None, 
        n=None
    ):
        assert_rank_and_shape_compatibility([
            reward, discount, reset, value, next_value, ratio
        ])
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
                norm_adv=norm_adv, 
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
                norm_adv=norm_adv, 
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
        event=None, 
        global_state, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
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
        hx = get_hx(idx, event)
        value = self.value(global_state, hx=hx)
        if next_obs is None:
            x = x[:, :-1]
            if idx is not None:
                idx = idx[:, :-1]
            if event is not None:
                event = event[:, :-1]
            next_value = value[:, 1:]
            value = value[:, :-1]
        else:
            with tape.stop_recording():
                assert state is None, 'unexpected states'
                if next_global_state is None:
                    next_global_state = next_obs
                next_x, _ = self.model.encode(next_global_state)
                next_value = self.value(next_x, hx=next_idx)
        hx = get_hx(idx, event)
        act_dist = self.policy(x, hx=hx, action_mask=action_mask)
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
                next_value=next_value, 
                ratio=ratio, 
                gamma=gamma, 
                lam=lam, 
                norm_adv=self.config.get('norm_adv', False), 
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

    def outer_loss(
        self, 
        *, 
        tape, 
        obs, 
        idx=None, 
        event=None, 
        hidden_state, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_hidden_state=None, 
        action, 
        old_value, 
        meta_reward, 
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
        terms = {}

        x, _ = self.model.encode(
            x=obs, 
            state=state, 
            mask=mask
        )
        tf.debugging.assert_equal(
            tf.gather(hidden_state, 0, axis=-2), 
            tf.gather(hidden_state, 1, axis=-2), 
        )
        hidden_state = tf.gather(hidden_state, [0], axis=2)
        value = self.meta_value(hidden_state)
        if next_hidden_state is None:
            x = x[:, :-1]
            if idx is not None:
                idx = idx[:, :-1]
            if event is not None:
                event = event[:, :-1]
            next_value = value[:, 1:]
            value = value[:, :-1]
        else:
            with tape.stop_recording():
                assert state is None, 'unexpected states'
                tf.debugging.assert_equal(
                    tf.gather(next_hidden_state, 0, axis=2), 
                    tf.gather(next_hidden_state, 1, axis=2), 
                )
                next_hidden_state = tf.gather(next_hidden_state, [0], axis=2)
                next_value = self.meta_value(next_hidden_state)
        hx = get_hx(idx, event)
        act_dist = self.policy(x, hx=hx, action_mask=action_mask)
        pi_logprob = act_dist.log_prob(action)
        assert_rank_and_shape_compatibility([pi_logprob, mu_logprob])
        log_ratio = pi_logprob - mu_logprob
        ratio = tf.exp(log_ratio)
        if sample_mask is not None:
            bool_mask = tf.cast(sample_mask, tf.bool)
            ratio = tf.where(bool_mask, ratio, 1.)
            pi_logprob = tf.where(bool_mask, pi_logprob, 0.)
        joint_ratio = tf.math.reduce_prod(ratio, axis=2, keepdims=True)
        joint_pi_logprob = tf.math.reduce_sum(pi_logprob, axis=2, keepdims=True)
        assert joint_ratio.shape[-1] == 1, joint_ratio.shape
        assert joint_pi_logprob.shape[-1] == 1, joint_pi_logprob.shape
        # tf.debugging.assert_near(
        #     tf.where(tf.cast(reset, bool), 0., log_ratio), 0., 1e-5, 1e-5)

        with tape.stop_recording():
            v_target, advantage = self.compute_target_advantage(
                reward=tf.reduce_mean(reward, axis=-1, keepdims=True), 
                discount=tf.math.reduce_max(discount, axis=-1, keepdims=True), 
                reset=tf.gather(reset, [0], axis=-1), 
                value=value, 
                next_value=next_value, 
                ratio=joint_ratio, 
                gamma=gamma, 
                lam=lam, 
                norm_adv=self.config.get('norm_meta_adv', False)
            )

        pg_coef = self.model.meta('pg_coef', inner=False)
        loss_pg, loss_clip, raw_pg_loss, actor_loss, clip_frac = \
            rl_loss.ppo_loss(
                pg_coef=pg_coef, 
                advantage=advantage, 
                ratio=joint_ratio, 
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
            pg_loss=actor_loss, 
            clip_frac=clip_frac, 
        )
        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value,
            target=v_target, 
            old_value=None, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        meta_reward = tf.math.abs(meta_reward)
        raw_meta_reward_loss, meta_reward_loss = rl_loss.to_loss(
            meta_reward, 
            self.config.meta_reward_coef, 
            mask=mask, 
            n=n
        )
        loss = actor_loss + value_loss + meta_reward_loss
        
        terms.update(value_terms)
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma, 
            lam=lam, 
            pi_logprob=pi_logprob, 
            raw_meta_reward_loss=raw_meta_reward_loss,
            meta_reward_loss=meta_reward_loss,
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
        next_obs=None, 
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
