from jax import lax, nn
import jax.numpy as jnp

from core.elements.loss import Loss
from core.typing import AttrDict
from jax_tools import jax_loss, jax_math
from tools.utils import prefix_name
from .utils import compute_joint_stats, compute_values, compute_policy


class MetaLoss(Loss):
    def loss(
        self, 
        theta, 
        eta, 
        data, 
        name='train', 
        use_meta=None, 
        use_dice=None, 
    ):
        stats = self.modules.meta_params(
            eta.meta_params, self.rng, inner=use_meta
        )

        if data.action_mask is not None:
            stats.n_avail_actions = jnp.sum(data.action_mask, -1)
        if data.sample_mask is not None:
            n_alive_units = jnp.sum(data.sample_mask, -1)
            stats.n_alive_units = n_alive_units

        data.value, next_value = compute_values(
            self.modules.value, 
            theta.value, 
            self.rng, 
            data.global_state, 
            data.next_global_state, 
            sid=data.sid, 
            next_sid=data.next_sid, 
            idx=data.idx,  
            next_idx=data.next_idx, 
            event=data.event, 
            next_event=data.next_event
        )

        act_dist, data.pi_logprob, stats.log_ratio, data.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                self.rng, 
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
                action_mask=data.action_mask
            )
        stats = record_policy_stats(data, stats, act_dist)

        v_target, stats.raw_adv = jax_loss.compute_target_advantage(
            config=self.config, 
            reward=data.rl_reward, 
            discount=data.rl_discount, 
            reset=data.rl_reset, 
            value=lax.stop_gradient(data.value), 
            next_value=next_value, 
            ratio=data.ratio, 
            gamma=stats.gamma, 
            lam=stats.lam, 
        )
        data.v_target = lax.stop_gradient(v_target) \
            if self.config.stop_target_grads else v_target
        stats = record_target_adv(data, stats)
        if self.config.norm_adv:
            data.advantage = jax_math.standard_normalization(
                stats.raw_adv, 
                zero_center=self.config.get('zero_center', True), 
                mask=data.sample_mask, 
                n=data.n, 
                epsilon=self.config.get('epsilon', 1e-8), 
                clip=self.config.adv_clip
            )
        else:
            data.advantage = stats.raw_adv

        actor_loss, stats = pg_loss(
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
        
        if data.reward is not None:
            if self.config.joint_objective:
                outer_value, _, _, outer_v_target, _ = compute_joint_stats(
                    config=self.config, 
                    func=self.modules.outer_value, 
                    params=theta.outer_value, 
                    rng=self.rng, 
                    hidden_state=data.hidden_state, 
                    next_hidden_state=data.next_hidden_state, 
                    sid=data.sid, 
                    next_sid=data.next_sid, 
                    reward=data.reward, 
                    discount=data.discount, 
                    reset=data.reset, 
                    ratio=data.ratio, 
                    pi_logprob=data.pi_logprob, 
                    gamma=stats.gamma, 
                    lam=stats.lam, 
                    sample_mask=data.sample_mask
                )
            else:
                outer_value, next_outer_value = compute_values(
                    self.modules.outer_value, 
                    theta.outer_value, 
                    self.rng, 
                    data.global_state, 
                    data.next_global_state, 
                    sid=data.sid, 
                    next_sid=data.next_sid, 
                    idx=data.idx, 
                    next_idx=data.next_idx, 
                )

                outer_v_target, _ = jax_loss.compute_target_advantage(
                    config=self.config, 
                    reward=data.reward, 
                    discount=data.discount, 
                    reset=data.reset, 
                    value=outer_value, 
                    next_value=next_outer_value, 
                    ratio=data.ratio, 
                    gamma=stats.gamma, 
                    lam=stats.lam, 
                )
            outer_data = AttrDict()
            outer_data.value = outer_value
            outer_data.v_target = outer_v_target
            outer_value_loss, outer_value_stats = vf_loss(
                self.config, 
                outer_data, 
                stats, 
                AttrDict()
            )
            loss = loss + outer_value_loss
            outer_value_stats = prefix_name(
                outer_value_stats, 
                'train/outer' if name is None else f'{name}/outer'
            )
            stats.update(outer_value_stats)
            stats.total_loss = loss
            
        stats = prefix_name(stats, name)

        return loss, stats

    def outer_loss(
        self, 
        theta, 
        data, 
        name='meta', 
    ):
        stats = self.modules.meta_params(
            {}, self.rng, inner=False
        )
        
        act_dist, data.pi_logprob, stats.log_ratio, data.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                self.rng, 
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
                action_mask=data.action_mask
            )
        stats = record_policy_stats(data, stats, act_dist)

        if self.config.joint_objective:
            data.value, data.joint_ratio, data.joint_pi_logprob, \
                data.v_target, stats.raw_adv = compute_joint_stats(
                    config=self.config, 
                    func=self.modules.outer_value, 
                    params=theta.outer_value, 
                    rng=self.rng, 
                    hidden_state=data.hidden_state, 
                    next_hidden_state=data.next_hidden_state, 
                    sid=data.sid, 
                    next_sid=data.next_sid, 
                    reward=data.reward, 
                    discount=data.discount, 
                    reset=data.reset, 
                    ratio=data.ratio, 
                    pi_logprob=data.pi_logprob, 
                    gamma=stats.gamma, 
                    lam=stats.lam, 
                    sample_mask=data.sample_mask
                )
            stats = record_target_adv(data, stats)
            if self.config.norm_meta_adv:
                data.advantage = jax_math.standard_normalization(
                    stats.raw_adv, 
                    zero_center=self.config.get('zero_center', True), 
                    epsilon=self.config.get('epsilon', 1e-8), 
                    clip=self.config.adv_clip
                )
            else:
                data.advantage = stats.raw_adv

            actor_loss, stats = joint_pg_loss(
                self.config, 
                data, 
                stats, 
                act_dist, 
            )
        else:
            data.value, next_value = compute_values(
                self.modules.outer_value, 
                theta.outer_value, 
                self.rng, 
                data.global_state, 
                data.next_global_state, 
                sid=data.sid, 
                next_sid=data.next_sid, 
                idx=data.idx,  
                next_idx=data.next_idx, 
            )

            data.v_target, stats.raw_adv = jax_loss.compute_target_advantage(
                config=self.config, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                value=lax.stop_gradient(data.value), 
                next_value=next_value, 
                ratio=data.ratio, 
                gamma=stats.gamma, 
                lam=stats.lam, 
            )
            stats = record_target_adv(data, stats)
            if self.config.norm_meta_adv:
                data.advantage = jax_math.standard_normalization(
                    stats.raw_adv, 
                    zero_center=self.config.get('zero_center', True), 
                    mask=data.sample_mask, 
                    n=data.n, 
                    epsilon=self.config.get('epsilon', 1e-8), 
                    clip=self.config.adv_clip
                )
            else:
                data.advantage = stats.raw_adv

            actor_loss, stats = pg_loss(
                self.config, 
                data, 
                stats, 
                act_dist=act_dist, 
                use_dice=False
            )

        raw_meta_reward_loss, meta_reward_loss = jax_loss.to_loss(
            lax.abs(data.meta_reward), 
            self.config.meta_reward_coef, 
            mask=data.sample_mask, 
            n=data.n
        )
        loss = actor_loss + meta_reward_loss
        stats.raw_meta_reward_loss = raw_meta_reward_loss
        stats.meta_reward_loss = meta_reward_loss
        stats.loss = loss

        stats = prefix_name(stats, name)
        
        return loss, stats


def create_loss(config, model, name='zero'):
    loss = MetaLoss(config=config, model=model, name=name)

    return loss


def pg_loss(
    config, 
    data, 
    stats, 
    act_dist, 
    use_dice=None
):
    use_dice = config.use_dice and use_dice
    
    if not config.get('policy_life_mask', True):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    if use_dice:
        dice_op = jax_loss.dice(
            data.pi_logprob, 
            axis=config.dice_axis, 
            lam=config.dice_lam
        )
    else:
        dice_op = data.pi_logprob
    stats.dice_op = dice_op

    if config.pg_type == 'pg':
        raw_pg_loss = jax_loss.pg_loss(
            advantage=data.advantage, 
            logprob=dice_op, 
            ratio=data.ratio, 
        )
    elif config.pg_type == 'ppo':
        if use_dice:
            ppo_pg_loss, ppo_clip_loss, raw_pg_loss = \
                jax_loss.high_order_ppo_loss(
                    advantage=data.advantage, 
                    ratio=data.ratio, 
                    dice_op=dice_op, 
                    clip_range=config.ppo_clip_range, 
                )
        else:
            ppo_pg_loss, ppo_clip_loss, raw_pg_loss = \
                jax_loss.ppo_loss(
                    advantage=data.advantage, 
                    ratio=data.ratio, 
                    clip_range=config.ppo_clip_range, 
                )
        stats.ppo_pg_loss = ppo_pg_loss
        stats.ppo_clip_loss = ppo_clip_loss
    else:
        raise NotImplementedError
    scaled_pg_loss, pg_loss = jax_loss.to_loss(
        raw_pg_loss, 
        stats.pg_coef, 
        mask=sample_mask, 
        n=data.n
    )
    stats.raw_pg_loss = raw_pg_loss
    stats.scaled_pg_loss = scaled_pg_loss
    stats.pg_loss = pg_loss

    entropy = act_dist.entropy()
    scaled_entropy_loss, entropy_loss = jax_loss.entropy_loss(
        entropy_coef=stats.entropy_coef, 
        entropy=entropy, 
        mask=sample_mask, 
        n=data.n
    )
    stats.entropy = entropy
    stats.scaled_entropy_loss = scaled_entropy_loss
    stats.entropy_loss = entropy_loss

    loss = pg_loss + entropy_loss
    stats.actor_loss = loss

    clip_frac = jax_math.mask_mean(
        lax.abs(data.ratio - 1.) > config.get('ppo_clip_range', .2), 
        sample_mask, data.n)
    stats.clip_frac = clip_frac

    return loss, stats

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

    value_loss_type = config.value_loss
    if value_loss_type == 'huber':
        raw_value_loss = jax_loss.huber_loss(
            data.value, 
            data.v_target, 
            threshold=config.huber_threshold
        )
    elif value_loss_type == 'mse':
        raw_value_loss = .5 * (data.value - data.v_target)**2
    elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
        raw_value_loss, new_stats.v_clip_frac = jax_loss.clipped_value_loss(
            data.value, 
            data.v_target, 
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


def joint_pg_loss(
    config, 
    data, 
    stats, 
    act_dist
):
    if not config.get('policy_life_mask', True):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    loss_pg, loss_clip, raw_pg_loss = \
        jax_loss.joint_ppo_loss(
            advantage=data.advantage, 
            ratio=data.joint_ratio, 
            clip_range=config.ppo_clip_range, 
            mask=sample_mask, 
            n=data.n, 
        )
    stats.pg_loss = loss_pg
    stats.clip_loss = loss_clip

    scaled_pg_loss, pg_loss = jax_loss.to_loss(
        raw_pg_loss, 
        stats.pg_coef, 
        mask=sample_mask, 
        n=data.n
    )
    stats.raw_pg_loss = raw_pg_loss
    stats.scaled_pg_loss = scaled_pg_loss
    stats.pg_loss = pg_loss

    entropy = act_dist.entropy()
    scaled_entropy_loss, entropy_loss = jax_loss.entropy_loss(
        entropy_coef=stats.entropy_coef, 
        entropy=entropy, 
        mask=sample_mask, 
        n=data.n
    )
    stats.entropy = entropy
    stats.scaled_entropy_loss = scaled_entropy_loss
    stats.entropy_loss = entropy_loss

    loss = pg_loss + entropy_loss
    stats.actor_loss = loss

    clip_frac = jax_math.mask_mean(
        lax.abs(data.ratio - 1.) > config.get('ppo_clip_range', .2), 
        sample_mask, data.n)
    stats.clip_frac = clip_frac

    return loss, stats


def record_policy_stats(data, stats, act_dist):
    stats.diff_frac = jax_math.mask_mean(
        lax.abs(data.pi_logprob - data.mu_logprob) > 1e-5, 
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

def record_target_adv(data, stats):
    stats.explained_variance = jax_math.explained_variance(data.v_target, data.value)
    stats.v_target_unit_std = jnp.std(data.v_target, axis=-1)
    stats.raw_adv_unit_std = jnp.std(stats.raw_adv, axis=-1)
    return stats
