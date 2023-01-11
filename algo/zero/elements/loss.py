from jax import lax, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_loss, jax_math, jax_utils
from tools.rms import denormalize, normalize
from tools.utils import prefix_name
from .utils import compute_values, compute_policy


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data, 
        name='train', 
    ):
        rngs = random.split(rng, 2)
        stats = dict2AttrDict(self.config.stats, to_copy=True)

        if data.action_mask is not None:
            stats.n_avail_actions = jnp.sum(data.action_mask, -1)
        if data.sample_mask is not None:
            stats.n_alive_units = jnp.sum(data.sample_mask, -1)

        stats.value, next_value = compute_values(
            self.modules.value, 
            theta.value, 
            rngs[0], 
            data.global_state, 
            data.next_global_state, 
            data.state_reset, 
            None if data.state is None else data.state.value, 
            bptt=self.config.vrnn_bptt, 
            seq_axis=1, 
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                theta.policy, 
                rngs[1], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                data.state_reset[:, :-1] if 'state_reset' in data else None, 
                None if data.state is None else data.state.policy, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                bptt=self.config.prnn_bptt, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)
        
        if 'advantage' in data:
            stats.advantage = data.pop('advantage')
            stats.v_target = data.pop('v_target')
        else:
            if self.config.popart:
                value = lax.stop_gradient(denormalize(
                    stats.value, data.popart_mean, data.popart_std))
                next_value = denormalize(next_value, data.popart_mean, data.popart_std)
            else:
                value = lax.stop_gradient(stats.value)

            v_target, stats.raw_adv = jax_loss.compute_target_advantage(
                config=self.config, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                value=value, 
                next_value=next_value, 
                ratio=lax.stop_gradient(stats.ratio), 
                gamma=stats.gamma, 
                lam=stats.lam, 
                axis=1
            )
            stats.v_target = lax.stop_gradient(v_target)
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
            stats.advantage = lax.stop_gradient(stats.advantage)

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )

        kl_stats = dict(
            logp=stats.pi_logprob, 
            logq=data.mu_logprob, 
            sample_prob=data.mu_logprob, 
            p_logits=stats.pi_logits, 
            q_logits=data.mu_logits, 
            p_loc=stats.pi_loc,  
            p_scale=stats.pi_scale, 
            q_loc=data.mu_loc,  
            q_scale=data.mu_scale, 
            action_mask=data.action_mask, 
            sample_mask=data.sample_mask, 
            n=data.n
        )
        stats.kl, stats.raw_kl_loss, stats.kl_loss = jax_loss.compute_kl(
            kl_type=self.config.kl_type, 
            kl_coef=self.config.kl_coef, 
            **kl_stats
        )
        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        loss = actor_loss + value_loss + stats.kl_loss
        stats.loss = loss

        return loss, stats

    def value_loss(
        self, 
        theta, 
        rng, 
        policy_theta, 
        data, 
        name='train/value', 
    ):
        rngs = random.split(rng, 2)
        stats = dict2AttrDict(self.config.stats, to_copy=True)

        if data.action_mask is not None:
            stats.n_avail_actions = jnp.sum(data.action_mask, -1)
        if data.sample_mask is not None:
            stats.n_alive_units = jnp.sum(data.sample_mask, -1)

        stats.value, next_value = compute_values(
            self.modules.value, 
            theta, 
            rngs[0], 
            data.global_state, 
            data.next_global_state, 
            data.state_reset, 
            None if data.state is None else data.state.value, 
            bptt=self.config.vrnn_bptt, 
            seq_axis=1, 
        )

        _, _, _, ratio = compute_policy(
            self.modules.policy, 
            policy_theta, 
            rngs[1], 
            data.obs, 
            data.next_obs, 
            data.action, 
            data.mu_logprob, 
            data.state_reset[:, :-1] if 'state_reset' in data else None, 
            None if data.state is None else data.state.policy, 
            action_mask=data.action_mask, 
            next_action_mask=data.next_action_mask, 
            bptt=self.config.prnn_bptt, 
            seq_axis=1, 
        )

        if 'advantage' in data:
            stats.raw_adv = data.pop('advantage')
            stats.v_target = data.pop('v_target')
        else:
            if self.config.popart:
                value = lax.stop_gradient(denormalize(
                    stats.value, data.popart_mean, data.popart_std))
                next_value = denormalize(next_value, data.popart_mean, data.popart_std)
            else:
                value = lax.stop_gradient(stats.value)

            v_target, stats.raw_adv = jax_loss.compute_target_advantage(
                config=self.config, 
                reward=data.reward, 
                discount=data.discount, 
                reset=data.reset, 
                value=value, 
                next_value=next_value, 
                ratio=lax.stop_gradient(ratio), 
                gamma=stats.gamma, 
                lam=stats.lam, 
                axis=1
            )
            stats.v_target = lax.stop_gradient(v_target)
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
        stats.advantage = lax.stop_gradient(stats.advantage)

        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        loss = value_loss

        return loss, stats

    def policy_loss(
        self, 
        theta, 
        rng, 
        data, 
        stats, 
        name='train/policy', 
    ):
        rngs = random.split(rng, 2)

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.modules.policy, 
                theta, 
                rngs[1], 
                data.obs, 
                data.next_obs, 
                data.action, 
                data.mu_logprob, 
                data.state_reset[:, :-1] if 'state_reset' in data else None, 
                None if data.state is None else data.state.policy, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                bptt=self.config.prnn_bptt, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )

        kl_stats = dict(
            logp=stats.pi_logprob, 
            logq=data.mu_logprob, 
            sample_prob=data.mu_logprob, 
            p_logits=stats.pi_logits, 
            q_logits=data.mu_logits, 
            p_loc=stats.pi_loc,  
            p_scale=stats.pi_scale, 
            q_loc=data.mu_loc,  
            q_scale=data.mu_scale, 
            action_mask=data.action_mask, 
            sample_mask=data.sample_mask, 
            n=data.n
        )
        stats.kl, stats.raw_kl_loss, stats.kl_loss = jax_loss.compute_kl(
            kl_type=self.config.kl_type, 
            kl_coef=self.config.kl_coef, 
            **kl_stats
        )
        loss = actor_loss + stats.kl_loss

        return loss, stats


def create_loss(config, model, name='zero'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def compute_actor_loss(
    config, 
    data, 
    stats, 
    act_dist, 
):
    if not config.get('policy_sample_mask', True):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    if config.pg_type == 'pg':
        raw_pg_loss = jax_loss.pg_loss(
            advantage=stats.advantage, 
            logprob=stats.pi_logprob, 
        )
    elif config.pg_type == 'ppo':
        ppo_pg_loss, ppo_clip_loss, raw_pg_loss = \
            jax_loss.ppo_loss(
                advantage=stats.advantage, 
                ratio=stats.ratio, 
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
        lax.abs(stats.ratio - 1.) > config.get('ppo_clip_range', .2), 
        sample_mask, data.n)
    stats.clip_frac = clip_frac

    return loss, stats


def compute_vf_loss(
    config, 
    data, 
    stats, 
):
    if config.get('value_sample_mask', False):
        sample_mask = data.sample_mask
    else:
        sample_mask = None
    
    if config.popart:
        v_target = normalize(
            stats.v_target, data.popart_mean, data.popart_std)
    else:
        v_target = stats.v_target
    stats.norm_v_target = v_target

    value_loss_type = config.value_loss
    if value_loss_type == 'huber':
        raw_value_loss = jax_loss.huber_loss(
            stats.value, 
            y=v_target, 
            threshold=config.huber_threshold
        )
    elif value_loss_type == 'mse':
        raw_value_loss = .5 * (stats.value - v_target)**2
    elif value_loss_type == 'clip' or value_loss_type == 'clip_huber':
        raw_value_loss, stats.v_clip_frac = jax_loss.clipped_value_loss(
            stats.value, 
            v_target, 
            data.value, 
            config.value_clip_range, 
            huber_threshold=config.huber_threshold, 
            mask=sample_mask, 
            n=data.n,
        )
    else:
        raise ValueError(f'Unknown value loss type: {value_loss_type}')
    stats.raw_value_loss = raw_value_loss
    scaled_value_loss, value_loss = jax_loss.to_loss(
        raw_value_loss, 
        coef=stats.value_coef, 
        mask=sample_mask, 
        n=data.n
    )
    
    stats.scaled_value_loss = scaled_value_loss
    stats.value_loss = value_loss

    return value_loss, stats


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
    stats.update(act_dist.get_stats(prefix='pi'))

    return stats
