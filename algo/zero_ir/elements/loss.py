import tensorflow as tf

from core.elements.loss import LossEnsemble
from utility import rl_loss, tf_utils
from .utils import compute_joint_stats, compute_values, compute_policy, prefix_name
from algo.zero.elements.loss import ValueLossImpl, POLossImpl


class Loss(ValueLossImpl, POLossImpl):
    def loss(
        self, 
        *, 
        tape, 
        obs, 
        idx=None, 
        event=None, 
        global_state=None, 
        hidden_state=None, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_global_state=None, 
        next_hidden_state=None, 
        action, 
        old_value, 
        rl_reward, 
        rl_discount, 
        rl_reset, 
        reward=None, 
        discount=None, 
        reset=None, 
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
        use_dice=None, 
        debug=True, 
    ):
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        gamma = self.model.meta('gamma', inner=use_meta)
        lam = self.model.meta('lam', inner=use_meta)

        if global_state is None:
            global_state = obs
        if next_global_state is None:
            next_global_state = next_obs
        value, next_value = compute_values(
            self.value, 
            global_state, 
            next_global_state, 
            idx, 
            next_idx, 
            event, 
            next_event
        )
        act_dist, pi_logprob, log_ratio, ratio = compute_policy(
            self.policy, obs, next_obs, action, mu_logprob, 
            idx, next_idx, event, next_event, action_mask
        )
        # tf.debugging.assert_near(
        #     tf.where(tf.cast(reset, bool), 0., log_ratio), 0., 1e-5, 1e-5)

        with tape.stop_recording():
            v_target, advantage = rl_loss.compute_target_advantage(
                config=self.config, 
                reward=rl_reward, 
                discount=rl_discount, 
                reset=rl_reset, 
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
            debug=debug, 
            use_dice=use_dice
        )
        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value,
            target=tf.stop_gradient(v_target) \
                if self.config.get('stop_target_grads', True) else v_target, 
            old_value=old_value, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=use_meta, 
            debug=debug
        )

        loss = actor_loss + value_loss
        
        terms = {**actor_terms, **value_terms}

        if reward is not None:
            if self.config.joint_objective:
                hidden_state = tf.gather(hidden_state, 0, axis=2)
                if next_hidden_state is not None:
                    next_hidden_state = tf.gather(next_hidden_state, 0, axis=2)
                outer_value, next_outer_value = compute_values(
                    self.outer_value, hidden_state, next_hidden_state
                )

                _, _, outer_v_target, _ = compute_joint_stats(
                    tape=tape, 
                    config=self.config, 
                    reward=reward, 
                    discount=discount, 
                    reset=reset, 
                    ratio=ratio, 
                    pi_logprob=pi_logprob, 
                    value=outer_value, 
                    next_value=next_outer_value, 
                    gamma=gamma, 
                    lam=lam, 
                    sample_mask=sample_mask
                )

                outer_value_loss, outer_value_terms = self._value_loss(
                    tape=tape, 
                    value=outer_value,
                    target=outer_v_target, 
                    old_value=None, 
                    name=name, 
                    use_meta=False, 
                    debug=debug
                )

                loss = loss + outer_value_loss
            else:
                outer_value, next_outer_value = compute_values(
                    self.outer_value, global_state, next_global_state
                )

                with tape.stop_recording():
                    outer_v_target, _ = rl_loss.compute_target_advantage(
                        config=self.config, 
                        reward=reward, 
                        discount=discount, 
                        reset=reset, 
                        value=outer_value, 
                        next_value=next_outer_value, 
                        ratio=ratio, 
                        gamma=gamma, 
                        lam=lam, 
                        norm_adv=self.config.get('norm_adv', False), 
                        mask=sample_mask, 
                        n=n
                    )

                outer_value_loss, outer_value_terms = self._value_loss(
                    tape=tape, 
                    value=outer_value,
                    target=outer_v_target, 
                    old_value=None, 
                    name=name, 
                    use_meta=False, 
                    debug=debug
                )

                loss = loss + outer_value_loss

            terms.update(outer_value_terms)

        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma, 
            lam=lam, 
            pi_logprob=pi_logprob, 
            ratio=ratio, 
            approx_kl=.5 * tf_utils.reduce_mean((log_ratio)**2, sample_mask, n), 
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
        global_state=None, 
        hidden_state, 
        next_obs=None, 
        next_idx=None, 
        next_event=None, 
        next_global_state=None, 
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
        debug=True, 
    ):
        n = None if sample_mask is None else tf.reduce_sum(sample_mask)
        gamma = self.model.meta('gamma', inner=False)
        lam = self.model.meta('lam', inner=False)
        terms = {}

        tf.debugging.assert_equal(
            tf.gather(hidden_state, 0, axis=-2), 
            tf.gather(hidden_state, 1, axis=-2), 
        )
        hidden_state = tf.gather(hidden_state, 0, axis=2)
        if next_hidden_state is not None:
            next_hidden_state = tf.gather(next_hidden_state, 0, axis=2)
        value, next_value = compute_values(
            self.outer_value, hidden_state, next_hidden_state
        )
        act_dist, pi_logprob, log_ratio, ratio = compute_policy(
            self.policy, obs, next_obs, action, mu_logprob, 
            idx, next_idx, event, next_event, action_mask
        )

        joint_ratio, joint_pi_logprob, _, advantage = compute_joint_stats(
            tape=tape, 
            config=self.config, 
            reward=reward, 
            discount=discount, 
            reset=reset, 
            ratio=ratio, 
            pi_logprob=pi_logprob, 
            value=value, 
            next_value=next_value, 
            gamma=gamma, 
            lam=lam, 
            sample_mask=sample_mask
        )
        
        pg_coef = self.model.meta('pg_coef', inner=False)
        entropy_coef = self.model.meta('entropy_coef', inner=False)

        loss_pg, loss_clip, raw_pg_loss, pg_loss, clip_frac = \
            rl_loss.joint_ppo_loss(
                pg_coef=pg_coef, 
                advantage=advantage, 
                ratio=ratio, 
                clip_range=self.config.ppo_clip_range, 
                mask=sample_mask, 
                n=n, 
            )
        entropy = act_dist.entropy()
        raw_entropy_loss, entropy_loss = rl_loss.entropy_loss(
            entropy_coef=entropy_coef, 
            entropy=entropy, 
            mask=sample_mask, 
            n=n
        )
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            loss_pg=loss_pg, 
            loss_clip=loss_clip, 
            raw_pg_loss=raw_pg_loss, 
            pg_loss=pg_loss, 
            clip_frac=clip_frac, 
            raw_entropy_loss=raw_entropy_loss, 
            entropy_loss=entropy_loss,
        )

        meta_reward = tf.math.abs(meta_reward)
        raw_meta_reward_loss, meta_reward_loss = rl_loss.to_loss(
            meta_reward, 
            self.config.meta_reward_coef, 
            mask=mask, 
            n=n
        )
        meta_loss = pg_loss + entropy_loss + meta_reward_loss
        
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma, 
            lam=lam, 
            joint_ratio=joint_ratio, 
            joint_pi_logprob=joint_pi_logprob, 
            raw_meta_reward_loss=raw_meta_reward_loss,
            meta_reward_loss=meta_reward_loss,
            approx_kl=.5 * tf_utils.reduce_mean((log_ratio)**2, sample_mask, n), 
            meta_loss=meta_loss, 
        )

        terms = prefix_name(terms, name)

        return meta_loss, terms


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
