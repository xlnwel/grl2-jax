from jax import lax, nn, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import AttrDict, dict2AttrDict
from jax_tools import jax_assert, jax_loss, jax_math, jax_utils
from tools.utils import prefix_name
from .utils import compute_joint_stats, compute_values, compute_policy, individual_pg_loss, joint_pg_loss


class MetaLoss(LossBase):
    def loss(
        self, 
        theta, 
        eta, 
        rng, 
        data, 
        name='theta', 
        use_meta=None, 
        use_dice=None, 
    ):
        rngs = random.split(rng, 3)
        stats = self.modules.meta_params(
            eta.meta_params, rngs[0], inner=use_meta
        )

        if data.action_mask is not None:
            stats.n_avail_actions = jnp.sum(data.action_mask, -1)
        if data.sample_mask is not None:
            stats.n_alive_units = jnp.sum(data.sample_mask, -1)

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                rngs[2], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx, 
                next_idx=data.next_idx, 
                event=data.event, 
                next_event=data.next_event, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        if 'event_discount' in data:
            rl_discount = data.event_discount
            rl_reset = None
            rl_discount, _ = jax_utils.split_data(rl_discount)
        else:
            rl_discount = data.discount
            rl_reset = data.reset
        if self.config.joint_theta_objective:
            stats.value, stats.joint_ratio, stats.joint_pi_logprob, \
                stats.v_target, stats.raw_adv = compute_joint_stats(
                    config=self.config, 
                    func=self.modules.value, 
                    params=theta.value, 
                    rng=rngs[1], 
                    hidden_state=data.hidden_state, 
                    next_hidden_state=data.next_hidden_state, 
                    sid=data.sid, 
                    next_sid=data.next_sid, 
                    reward=data.rl_reward, 
                    discount=rl_discount, 
                    reset=rl_reset, 
                    ratio=stats.ratio, 
                    pi_logprob=stats.pi_logprob, 
                    gamma=stats.gamma, 
                    lam=stats.lam, 
                    sample_mask=data.sample_mask
                )
            stats.explained_variance = jax_math.explained_variance(
                stats.v_target, stats.value)

            if self.config.norm_adv:
                advantage = jax_math.standard_normalization(
                    stats.raw_adv, 
                    zero_center=self.config.get('zero_center', True), 
                    epsilon=self.config.get('epsilon', 1e-8), 
                    clip=self.config.adv_clip
                )
            else:
                advantage = stats.raw_adv
            stats.advantage = lax.stop_gradient(advantage)

            actor_loss, stats = joint_pg_loss(
                self.config, 
                data, 
                stats, 
                act_dist=act_dist, 
            )
        else:
            stats.value, next_value = compute_values(
                self.modules.value, 
                theta.value, 
                rngs[1], 
                data.global_state, 
                data.next_global_state, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx, 
                next_idx=data.next_idx, 
                event=data.event, 
                next_event=data.next_event, 
                seq_axis=1, 
            )

            stats.v_target, stats.raw_adv = jax_loss.compute_target_advantage(
                config=self.config, 
                reward=data.rl_reward, 
                discount=rl_discount, 
                reset=rl_reset, 
                value=lax.stop_gradient(stats.value), 
                next_value=next_value, 
                ratio=lax.stop_gradient(stats.ratio), 
                gamma=stats.gamma, 
                lam=stats.lam, 
                axis=1
            )
            if self.config.stop_target_grads:
                stats.v_target = lax.stop_gradient(stats.v_target)
            stats = record_target_adv(stats)

            if self.config.norm_adv:
                stats.advantage = jax_math.standard_normalization(
                    stats.raw_adv, 
                    zero_center=self.config.get('zero_center', True), 
                    mask=data.sample_mask, 
                    n=data.n, 
                    epsilon=self.config.get('epsilon', 1e-8), 
                )
            else:
                stats.advantage = stats.raw_adv

            actor_loss, stats = individual_pg_loss(
                self.config, 
                data, 
                stats, 
                act_dist=act_dist, 
                use_dice=use_dice
            )

        value_loss, stats = vf_loss(
            self.config, 
            data, 
            stats, 
        )
        loss = actor_loss + value_loss
        stats.loss = loss

        stats = prefix_name(stats, name)
        
        return loss, stats
    
    def phi_loss(
        self, 
        phi, 
        theta, 
        rng, 
        data, 
        name='phi', 
    ):
        stats = self.modules.meta_params(
            {}, None, inner=False
        )

        rngs = random.split(rng, 2)
        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                rngs[0], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx, 
                next_idx=data.next_idx, 
                event=data.event, 
                next_event=data.next_event, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        config = dict2AttrDict(self.config, to_copy=True)
        config.target_type = 'vtrace'
        if config.joint_objective:
            stats.value, _, _, v_target, _ = compute_joint_stats(
                config=config, 
                func=self.modules.outer_value, 
                params=phi.outer_value, 
                rng=rngs[1], 
                hidden_state=data.hidden_state, 
                next_hidden_state=data.next_hidden_state, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                ratio=stats.ratio, 
                pi_logprob=stats.pi_logprob, 
                gamma=stats.gamma, 
                lam=stats.lam, 
                sample_mask=data.sample_mask
            )
        else:
            stats.value, next_value = compute_values(
                self.modules.outer_value, 
                phi.outer_value, 
                rngs[1], 
                data.global_state, 
                data.next_global_state, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx, 
                next_idx=data.next_idx, 
            )

            v_target, _ = jax_loss.compute_target_advantage(
                config=config, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                value=stats.value, 
                next_value=next_value, 
                ratio=stats.ratio, 
                gamma=stats.gamma, 
                lam=stats.lam, 
            )
        stats.v_target = lax.stop_gradient(v_target)
        value_loss, stats = vf_loss(
            config, 
            data, 
            stats, 
        )
        stats.value_loss = value_loss

        stats = prefix_name(stats, name)

        return value_loss, stats

    def eta_loss(
        self, 
        theta, 
        phi, 
        rng, 
        data, 
        name='eta', 
    ):
        stats = self.modules.meta_params(
            {}, rng, inner=False
        )
        
        rngs = random.split(rng, 2)
        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                rngs[0], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx, 
                next_idx=data.next_idx, 
                event=data.event, 
                next_event=data.next_event, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        if self.config.joint_objective:
            stats.value, stats.joint_ratio, stats.joint_pi_logprob, \
                stats.v_target, stats.raw_adv = compute_joint_stats(
                    config=self.config, 
                    func=self.modules.outer_value, 
                    params=phi.outer_value, 
                    rng=rngs[1], 
                    hidden_state=data.hidden_state, 
                    next_hidden_state=data.next_hidden_state, 
                    sid=data.sid, 
                    next_sid=data.next_sid, 
                    reward=data.reward, 
                    discount=data.discount, 
                    reset=data.reset, 
                    ratio=stats.ratio, 
                    pi_logprob=stats.pi_logprob, 
                    gamma=stats.gamma, 
                    lam=stats.lam, 
                    sample_mask=data.sample_mask
                )
            stats.explained_variance = jax_math.explained_variance(
                stats.v_target, stats.value)
            if self.config.norm_meta_adv:
                advantage = jax_math.standard_normalization(
                    stats.raw_adv, 
                    zero_center=self.config.get('zero_center', True), 
                    epsilon=self.config.get('epsilon', 1e-8), 
                    clip=self.config.adv_clip
                )
            else:
                advantage = stats.raw_adv
            stats.advantage = lax.stop_gradient(advantage)

            actor_loss, stats = joint_pg_loss(
                self.config, 
                data, 
                stats, 
                act_dist=act_dist, 
            )
        else:
            stats.value, next_value = compute_values(
                self.modules.outer_value, 
                phi.outer_value, 
                rngs[1], 
                data.global_state, 
                data.next_global_state, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx,  
                next_idx=data.next_idx, 
                seq_axis=1, 
            )

            stats.v_target, stats.raw_adv = jax_loss.compute_target_advantage(
                config=self.config, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                value=stats.value, 
                next_value=next_value, 
                ratio=stats.ratio, 
                gamma=stats.gamma, 
                lam=stats.lam, 
                axis=1, 
            )
            stats = record_target_adv(stats)
            if self.config.norm_meta_adv:
                advantage = jax_math.standard_normalization(
                    stats.raw_adv, 
                    zero_center=self.config.get('zero_center', True), 
                    mask=data.sample_mask, 
                    n=data.n, 
                    epsilon=self.config.get('epsilon', 1e-8), 
                    clip=self.config.adv_clip
                )
            else:
                advantage = stats.raw_adv
            stats.advantage = lax.stop_gradient(advantage)

            actor_loss, stats = individual_pg_loss(
                self.config, 
                data, 
                stats, 
                act_dist=act_dist, 
                use_dice=False
            )

        stats.kl, stats.raw_kl_loss, kl_loss = jax_loss.compute_kl(
            kl_type=self.config.kl, 
            kl_coef=self.config.kl_coef, 
            logp=data.mu_logprob, 
            logq=stats.pi_logprob, 
            sample_prob=data.mu_logprob, 
            pi1=data.mu, 
            pi2=stats.pi,
            pi1_mean=data.mu_mean,  
            pi1_std=data.mu_std,  
            pi2_mean=stats.pi_mean,  
            pi2_std=stats.pi_std,  
            pi_mask=data.action_mask, 
            sample_mask=data.sample_mask, 
            n=data.n
        )
        stats.kl_loss = kl_loss

        main_reward, _ = jax_utils.split_data(data.main_reward, data.next_reward)
        meta_reward_loss = jnp.square(jnp.maximum(jnp.where(
            main_reward == 0, lax.abs(data.meta_reward), self.config.free_nats), 
            self.config.free_nats))
        stats.raw_meta_reward_loss, \
            stats.meta_reward_loss = jax_loss.to_loss(
                meta_reward_loss, 
                self.config.meta_reward_coef, 
                mask=data.sample_mask, 
                n=data.n
            )

        indicator_loss = jnp.where(main_reward == 0, data.meta_indicator, 0)
        stats.raw_meta_indicator_loss, \
            stats.meta_indicator_loss = jax_loss.to_loss(
                indicator_loss, 
                self.config.meta_indicator_coef, 
                mask=data.sample_mask, 
                n=data.n
            )

        loss = actor_loss + kl_loss + stats.meta_reward_loss + stats.meta_indicator_loss
        stats.loss = loss

        stats = prefix_name(stats, name)
        
        return loss, stats


def create_loss(config, model, name='zero'):
    loss = MetaLoss(config=config, model=model, name=name)

    return loss


def vf_loss(
    config, 
    data, 
    stats, 
    new_stats=None, 
):
    if new_stats is None:
        new_stats = stats
    if config.get('value_life_mask', False):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    jax_assert.assert_shape_compatibility([
        stats.value, stats.v_target
    ])
    value_loss_type = config.value_loss
    if value_loss_type == 'huber':
        raw_value_loss = jax_loss.huber_loss(
            stats.value, 
            stats.v_target, 
            threshold=config.huber_threshold
        )
    elif value_loss_type == 'mse':
        raw_value_loss = .5 * (stats.value - stats.v_target)**2
    elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
        raw_value_loss, new_stats.v_clip_frac = jax_loss.clipped_value_loss(
            stats.value, 
            stats.v_target, 
            data.old_value, 
            config.value_clip_range, 
            huber_threshold=config.huber_threshold, 
            mask=sample_mask, 
            n=data.n,
        )
    else:
        raise ValueError(f'Unknown value loss type: {value_loss_type}')
    new_stats.raw_v_loss = raw_value_loss
    scaled_value_loss, value_loss = jax_loss.to_loss(
        raw_value_loss, 
        coef=stats.value_coef, 
        mask=sample_mask, 
        n=data.n
    )
    
    new_stats.scaled_v_loss = scaled_value_loss
    new_stats.v_loss = value_loss

    return value_loss, new_stats


def record_target_adv(stats):
    stats.explained_variance = jax_math.explained_variance(
        stats.v_target, stats.value)
    stats.v_target_unit_std = jnp.std(stats.v_target, axis=-1)
    stats.raw_adv_unit_std = jnp.std(stats.raw_adv, axis=-1)
    return stats


def record_policy_stats(data, stats, act_dist):
    stats.diff_frac = jax_math.mask_mean(
        lax.abs(stats.pi_logprob - data.mu_logprob) > 1e-5, 
        data.sample_mask, data.n)
    stats.approx_kl = .5 * jax_math.mask_mean(
        (stats.log_ratio)**2, data.sample_mask, data.n)
    stats.approx_kl_max = jnp.max(.5 * (stats.log_ratio)**2)
    if data.mu is not None:
        stats.pi = nn.softmax(act_dist.logits)
        stats.diff_pi = stats.pi - data.mu
    elif data.mu_mean is not None:
        stats.pi_mean = act_dist.mu
        stats.diff_pi_mean = act_dist.mu - data.mu_mean
        stats.pi_std = act_dist.std
        stats.diff_pi_std = act_dist.std - data.mu_std

    return stats
