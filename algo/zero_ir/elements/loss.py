import tensorflow as tf

from core.elements.loss import Loss as LossBase, LossEnsemble
from utility import rl_loss
from utility.tf_utils import assert_rank_and_shape_compatibility, reduce_mean
from .utils import get_hx
from algo.zero.elements.loss import Loss as LossBase, split_data, prefix_name


class Loss(LossBase):
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
        value, next_value = self._compute_values(
            self.outer_value, 
            hidden_state, 
            next_hidden_state
        )
        obs, _ = split_data(obs, next_obs)
        idx, _ = split_data(idx, next_idx)
        event, _ = split_data(event, next_event)
        hx = get_hx(idx, event)
        act_dist = self.policy(obs, hx=hx, action_mask=action_mask)
        pi_logprob = act_dist.log_prob(action)
        assert_rank_and_shape_compatibility([pi_logprob, mu_logprob])
        log_ratio = pi_logprob - mu_logprob
        ratio = tf.exp(log_ratio)
        if sample_mask is not None:
            bool_mask = tf.cast(sample_mask, tf.bool)
            ratio = tf.where(bool_mask, ratio, 1.)
            pi_logprob = tf.where(bool_mask, pi_logprob, 0.)
        joint_ratio = tf.math.reduce_prod(ratio, axis=-1)
        joint_pi_logprob = tf.math.reduce_sum(pi_logprob, axis=-1)

        with tape.stop_recording():
            v_target, advantage = self.compute_target_advantage(
                reward=tf.reduce_mean(reward, axis=-1), 
                discount=tf.math.reduce_max(discount, axis=-1), 
                reset=tf.gather(reset, 0, axis=-1), 
                value=value, 
                next_value=next_value, 
                ratio=joint_ratio, 
                gamma=gamma, 
                lam=lam, 
                norm_adv=self.config.get('norm_meta_adv', False)
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
        value_loss, value_terms = self._value_loss(
            tape=tape, 
            value=value,
            target=v_target, 
            old_value=None, 
            sample_mask=sample_mask, 
            n=n, 
            name=name, 
            use_meta=False, 
            debug=debug
        )

        meta_reward = tf.math.abs(meta_reward)
        raw_meta_reward_loss, meta_reward_loss = rl_loss.to_loss(
            meta_reward, 
            self.config.meta_reward_coef, 
            mask=mask, 
            n=n
        )
        plain_loss = value_loss + meta_reward_loss
        meta_loss = pg_loss + entropy_loss
        
        terms.update(value_terms)
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
            approx_kl=.5 * reduce_mean((log_ratio)**2, sample_mask, n), 
            plain_loss=plain_loss, 
            meta_loss=meta_loss, 
        )

        terms = prefix_name(terms, name)

        return plain_loss, meta_loss, terms


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
