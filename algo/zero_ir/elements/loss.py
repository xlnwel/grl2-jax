import tensorflow as tf

from tf_core.elements.loss import Loss as LossBase, LossEnsemble
from tf_tools import tf_utils, rl_utils, rl_loss
from .utils import compute_joint_stats, compute_values, compute_policy, prefix_name
from algo.zero.elements.loss import ValueLossImpl, POLossImpl


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
        debug=True, 
        use_dice=None
    ):
        use_dice = self.config.use_dice and use_dice
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

        if use_dice:
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
                ratio=ratio, 
                mask=sample_mask, 
                n=n, 
            )
        elif self.config.pg_type == 'ppo':
            if use_dice:
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
            else:
                loss_pg, loss_clip, raw_pg_loss, pg_loss, clip_frac = \
                    rl_loss.ppo_loss(
                        pg_coef=pg_coef, 
                        advantage=advantage, 
                        ratio=ratio, 
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
                ev = tf_utils.explained_variance(target, value)
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
            idx=idx, 
            next_idx=next_idx, 
            event=event, 
            next_event=next_event
        )
        act_dist, pi_logprob, log_ratio, ratio = compute_policy(
            self.policy, obs, next_obs, action, mu_logprob, 
            idx, next_idx, event, next_event, action_mask
        )
        # tf.print('reward', rl_reward[0, -1, -1])
        # tf.print('mu logprob', mu_logprob[0, -1, -1])
        # tf.print('pi logprob', pi_logprob[0, -1, -1])
        # tf.print('max diff', tf.math.reduce_max(pi_logprob - mu_logprob))
        # tf.debugging.assert_near(pi_logprob, mu_logprob, 1e-5, 1e-5)

        with tape.stop_recording():
            v_target, raw_adv = rl_loss.compute_target_advantage(
                config=self.config, 
                reward=reward, 
                discount=discount, 
                reset=reset, 
                value=value, 
                next_value=next_value, 
                ratio=ratio, 
                gamma=gamma, 
                lam=lam, 
            )
            adv = rl_utils.normalize_adv(
                self.config, 
                raw_adv, 
                norm_adv=self.config.get('norm_adv', False), 
                sample_mask=sample_mask, 
                n=n
            )

        actor_loss, actor_terms = self._pg_loss(
            tape=tape, 
            act_dist=act_dist, 
            advantage=adv, 
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

        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            raw_advantage=raw_adv, 
            gamma=gamma, 
            lam=lam, 
            pi_logprob=pi_logprob, 
            ratio=ratio, 
            approx_kl=.5 * tf_utils.reduce_mean((log_ratio)**2, sample_mask, n), 
            approx_kl_max=.5 * tf.math.reduce_max((log_ratio)**2), 
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


class MetaLoss(ValueLossImpl, POLossImpl):
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
            idx=idx, 
            next_idx=next_idx, 
            event=event, 
            next_event=next_event
        )
        act_dist, pi_logprob, log_ratio, ratio = compute_policy(
            self.policy, obs, next_obs, action, mu_logprob, 
            idx=idx, next_idx=next_idx, event=event, 
            next_event=next_event, 
            action_mask=action_mask
        )
        # tf.print('reward', rl_reward[0, -1, -1])
        # tf.print('mu logprob', mu_logprob[0, -1, -1])
        # tf.print('pi logprob', pi_logprob[0, -1, -1])
        # tf.print('max diff', tf.math.reduce_max(pi_logprob - mu_logprob))
        # tf.debugging.assert_near(pi_logprob, mu_logprob, 1e-5, 1e-5)

        with tape.stop_recording():
            v_target, raw_adv = rl_loss.compute_target_advantage(
                config=self.config, 
                reward=rl_reward, 
                discount=rl_discount, 
                reset=rl_reset, 
                value=value, 
                next_value=next_value, 
                ratio=ratio, 
                gamma=gamma, 
                lam=lam, 
            )
            adv = rl_utils.normalize_adv(
                self.config, 
                raw_adv, 
                norm_adv=self.config.get('norm_adv', False), 
                sample_mask=sample_mask, 
                n=n
            )

        actor_loss, actor_terms = self._pg_loss(
            tape=tape, 
            act_dist=act_dist, 
            advantage=adv, 
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

        # if reward is not None:
        #     if self.config.joint_objective:
        #         hidden_state = tf.gather(hidden_state, 0, axis=2)
        #         if next_hidden_state is not None:
        #             next_hidden_state = tf.gather(next_hidden_state, 0, axis=2)
        #         outer_value, next_outer_value = compute_values(
        #             self.outer_value, hidden_state, next_hidden_state
        #         )

        #         _, _, outer_v_target, _ = compute_joint_stats(
        #             tape=tape, 
        #             config=self.config, 
        #             reward=reward, 
        #             discount=discount, 
        #             reset=reset, 
        #             ratio=ratio, 
        #             pi_logprob=pi_logprob, 
        #             value=outer_value, 
        #             next_value=next_outer_value, 
        #             gamma=gamma, 
        #             lam=lam, 
        #             sample_mask=sample_mask
        #         )
        #         outer_value_loss, outer_value_terms = self._value_loss(
        #             tape=tape, 
        #             value=outer_value,
        #             target=outer_v_target, 
        #             old_value=None, 
        #             name=None, 
        #             use_meta=False, 
        #             debug=debug, 
        #         )
        #     else:
        #         outer_value, next_outer_value = compute_values(
        #             self.outer_value, 
        #             global_state, 
        #             next_global_state, 
        #             idx=idx, 
        #             next_idx=next_idx, 
        #             event=event, 
        #             next_event=next_event
        #         )

        #         with tape.stop_recording():
        #             outer_v_target, _ = rl_loss.compute_target_advantage(
        #                 config=self.config, 
        #                 reward=reward, 
        #                 discount=discount, 
        #                 reset=reset, 
        #                 value=outer_value, 
        #                 next_value=next_outer_value, 
        #                 ratio=ratio, 
        #                 gamma=gamma, 
        #                 lam=lam, 
        #             )
        #         outer_value_loss, outer_value_terms = self._value_loss(
        #             tape=tape, 
        #             value=outer_value,
        #             target=outer_v_target, 
        #             old_value=None, 
        #             sample_mask=sample_mask, 
        #             n=n, 
        #             name=None, 
        #             use_meta=False, 
        #             debug=debug, 
        #         )
        #     loss = loss + outer_value_loss
        #     outer_value_terms = prefix_name(outer_value_terms, 'train/outer')
        #     terms.update(outer_value_terms)
            
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma, 
            lam=lam, 
            raw_advantage=raw_adv, 
            pi_logprob=pi_logprob, 
            ratio=ratio, 
            approx_kl=.5 * tf_utils.reduce_mean((log_ratio)**2, sample_mask, n), 
            approx_kl_max=.5 * tf.math.reduce_max((log_ratio)**2), 
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

    def joint_loss(
        self, 
        tape, 
        gamma, 
        lam, 
        act_dist, 
        pi_logprob, 
        ratio, 
        hidden_state, 
        next_hidden_state, 
        reward, 
        discount, 
        reset, 
        sample_mask, 
        n, 
        name, 
        use_meta, 
        use_dice, 
        debug, 
    ):
        terms = {}
        pg_coef = self.model.meta('pg_coef', inner=False)
        entropy_coef = self.model.meta('entropy_coef', inner=False)

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

        joint_ratio, joint_pi_logprob, _, raw_adv = compute_joint_stats(
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
        with tape.stop_recording():
            adv = rl_utils.normalize_adv(
                self.config, 
                raw_adv, 
                norm_adv=self.config.get('norm_meta_adv', False), 
                sample_mask=sample_mask, 
                n=n
            )

        loss_pg, loss_clip, raw_pg_loss, pg_loss, clip_frac = \
            rl_loss.joint_ppo_loss(
                pg_coef=pg_coef, 
                advantage=adv, 
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
            raw_advantage=raw_adv, 
            advantage=adv, 
            joint_ratio=joint_ratio, 
            joint_pi_logprob=joint_pi_logprob, 
            loss_pg=loss_pg, 
            loss_clip=loss_clip, 
            raw_pg_loss=raw_pg_loss, 
            pg_loss=pg_loss, 
            clip_frac=clip_frac, 
            raw_entropy_loss=raw_entropy_loss, 
            entropy_loss=entropy_loss,
        )
        actor_loss = pg_loss + entropy_loss
        
        return actor_loss, terms

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

        act_dist, pi_logprob, log_ratio, ratio = compute_policy(
            self.policy, obs, next_obs, action, mu_logprob, 
            idx=idx, next_idx=next_idx, event=event, 
            next_event=next_event, 
            action_mask=action_mask
        )

        if self.config.joint_objective:
            actor_loss, terms = self.joint_loss(
                tape, 
                gamma, 
                lam, 
                act_dist, 
                pi_logprob, 
                ratio, 
                hidden_state, 
                next_hidden_state, 
                reward, 
                discount, 
                reset, 
                sample_mask=sample_mask, 
                n=n, 
                name=None, 
                use_meta=False, 
                use_dice=False, 
                debug=debug, 
            )
        else:
            if global_state is None:
                global_state = obs
            if next_global_state is None:
                next_global_state = next_obs

            value, next_value = compute_values(
                self.outer_value, 
                global_state, 
                next_global_state, 
                idx=idx, 
                next_idx=next_idx, 
                event=event, 
                next_event=next_event
            )

            with tape.stop_recording():
                _, raw_adv = rl_loss.compute_target_advantage(
                    config=self.config, 
                    reward=reward, 
                    discount=discount, 
                    reset=reset, 
                    value=value, 
                    next_value=next_value, 
                    ratio=ratio, 
                    gamma=gamma, 
                    lam=lam, 
                )
                adv = rl_utils.normalize_adv(
                    self.config, 
                    raw_adv, 
                    norm_adv=self.config.get('norm_meta_adv', False), 
                    sample_mask=sample_mask, 
                    n=n
                )

            actor_loss, terms = self._pg_loss(
                tape=tape, 
                act_dist=act_dist, 
                advantage=adv, 
                ratio=ratio, 
                pi_logprob=pi_logprob, 
                mu=mu, 
                mu_mean=mu_mean, 
                mu_std=mu_std, 
                action_mask=action_mask, 
                sample_mask=sample_mask, 
                n=n, 
                name=name, 
                use_meta=False, 
                use_dice=False, 
                debug=debug, 
            )
            self.log_for_debug(
                tape, 
                terms, 
                debug=debug, 
                gamma=gamma, 
                lam=lam, 
                raw_advantage=raw_adv, 
            )

        meta_reward = tf.math.abs(meta_reward)
        raw_meta_reward_loss, meta_reward_loss = rl_loss.to_loss(
            meta_reward, 
            self.config.meta_reward_coef, 
            mask=mask, 
            n=n
        )
        meta_loss = actor_loss + meta_reward_loss
        
        self.log_for_debug(
            tape, 
            terms, 
            debug=debug, 
            gamma=gamma, 
            lam=lam, 
            raw_meta_reward_loss=raw_meta_reward_loss,
            meta_reward_loss=meta_reward_loss,
            approx_kl=.5 * tf_utils.reduce_mean((log_ratio)**2, sample_mask, n), 
            meta_loss=meta_loss, 
        )

        terms = prefix_name(terms, name)

        return meta_loss, terms


def create_loss(config, model, name='zero'):
    rl_loss = MetaLoss(config=config.rl, model=model.rl, name='rl')
    meta_loss = MetaLoss(config=config.meta, model=model.meta, name='meta')

    return LossEnsemble(
        config=config, 
        model=model, 
        components=dict(
            rl=rl_loss, 
            meta=meta_loss
        ), 
        name=name, 
    )
