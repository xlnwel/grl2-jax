from jax import lax, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_div, jax_loss, jax_math
from tools.rms import denormalize
from .utils import *


class Loss(LossBase):
    def loss(
        self, 
        theta, 
        rng, 
        data,
        teammate_log_ratio,
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
                self.model, 
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
            stats.raw_adv = data.pop('advantage')
            stats.v_target = data.pop('v_target')
        else:
            if self.config.popart:
                value = lax.stop_gradient(denormalize(
                    stats.value, data.popart_mean, data.popart_std))
                next_value = denormalize(
                    next_value, data.popart_mean, data.popart_std)
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
        stats.norm_adv, stats.advantage = norm_adv(
            self.config, 
            stats.raw_adv, 
            teammate_log_ratio, 
            sample_mask=data.sample_mask, 
            n=data.n, 
            epsilon=self.config.get('epsilon', 1e-5)
        )

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )

        stats = compute_regularization(
            stats, data, self.config.reg_type, self.config.pos_reg_coef, 
            self.config.rescaled_by_adv, self.config.lower_threshold)
        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        stats = summarize_adv_ratio(stats, data)
        loss = actor_loss + value_loss + stats.pos_reg_loss
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
            self.model, 
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
                next_value = denormalize(
                    next_value, data.popart_mean, data.popart_std)
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
        teammate_log_ratio, 
        name='train/policy', 
    ):
        rngs = random.split(rng, 2)
        stats.norm_adv, stats.advantage = norm_adv(
            self.config, 
            stats.raw_adv, 
            teammate_log_ratio, 
            sample_mask=data.sample_mask, 
            n=data.n, 
            epsilon=self.config.get('epsilon', 1e-5)
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.model, 
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

        stats = compute_regularization(
            stats, data, self.config.reg_type, self.config.pos_reg_coef,
            self.config.rescaled_by_adv, self.config.lower_threshold)
        stats = summarize_adv_ratio(stats, data)
        loss = actor_loss + stats.pos_reg_loss

        return loss, stats

    def lka_loss(
        self, 
        theta, 
        rng, 
        data,
        teammate_log_ratio,
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
            bptt=self.config.lka_vrnn_bptt, 
            seq_axis=1, 
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.model, 
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
                bptt=self.config.lka_prnn_bptt, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        if 'advantage' in data:
            stats.raw_adv = data.pop('advantage')
            stats.v_target = data.pop('v_target')
        else:
            if self.config.popart:
                value = lax.stop_gradient(denormalize(
                    stats.value, data.popart_mean, data.popart_std))
                next_value = denormalize(
                    next_value, data.popart_mean, data.popart_std)
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

        stats.norm_adv, stats.advantage = norm_adv(
            self.config, 
            stats.raw_adv, 
            teammate_log_ratio, 
            sample_mask=data.sample_mask, 
            n=data.n, 
            epsilon=self.config.get('epsilon', 1e-5)
        )

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )

        stats = compute_regularization(
            stats, data, self.config.lka_reg_type, self.config.pos_lka_reg_coef,
            self.config.rescaled_by_adv, self.config.lower_threshold)
        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        loss = actor_loss + value_loss + stats.pos_reg_loss
        stats.loss = loss

        return loss, stats

    def lka_value_loss(
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
            bptt=self.config.lka_vrnn_bptt, 
            seq_axis=1, 
        )

        _, _, _, ratio = compute_policy(
            self.model, 
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
                next_value = denormalize(
                    next_value, data.popart_mean, data.popart_std)
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

        value_loss, stats = compute_vf_loss(
            self.config, 
            data, 
            stats, 
        )
        loss = value_loss

        return loss, stats

    def lka_policy_loss(
        self, 
        theta, 
        rng, 
        data, 
        stats,
        teammate_log_ratio,
        name='train/policy', 
    ):
        rngs = random.split(rng, 2)
        stats.norm_adv, stats.advantage = norm_adv(
            self.config, 
            stats.raw_adv, 
            teammate_log_ratio, 
            sample_mask=data.sample_mask, 
            n=data.n, 
            epsilon=self.config.get('epsilon', 1e-5)
        )

        act_dist, stats.pi_logprob, stats.log_ratio, stats.ratio = \
            compute_policy(
                self.model, 
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
                bptt=self.config.lka_prnn_bptt, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        actor_loss, stats = compute_actor_loss(
            self.config, 
            data, 
            stats, 
            act_dist=act_dist, 
        )

        stats = compute_regularization(
            stats, data, self.config.lka_reg_type, self.config.pos_lka_reg_coef,
            self.config.rescaled_by_adv, self.config.lower_threshold)

        loss = actor_loss + stats.pos_reg_loss

        return loss, stats


def create_loss(config, model, name='happo'):
    loss = Loss(config=config, model=model, name=name)

    return loss


def summarize_adv_ratio(stats, data):
    stats.raw_adv_ratio_pp = jax_math.mask_mean(
        jnp.logical_and(stats.raw_adv > 0, stats.ratio > 1), 
        data.sample_mask, data.n)
    stats.raw_adv_ratio_pn = jax_math.mask_mean(
        jnp.logical_and(stats.raw_adv > 0, stats.ratio < 1), 
        data.sample_mask, data.n)
    # stats.raw_adv_ratio_np = jax_math.mask_mean(
    #     jnp.logical_and(stats.raw_adv < 0, stats.ratio > 1), 
    #     data.sample_mask, data.n)
    # stats.raw_adv_ratio_nn = jax_math.mask_mean(
    #     jnp.logical_and(stats.raw_adv < 0, stats.ratio < 1), 
    #     data.sample_mask, data.n)
    # stats.adv_ratio_pp = jax_math.mask_mean(
    #     jnp.logical_and(stats.advantage > 0, stats.ratio > 1), 
    #     data.sample_mask, data.n)
    # stats.adv_ratio_pn = jax_math.mask_mean(
    #     jnp.logical_and(stats.advantage > 0, stats.ratio < 1), 
    #     data.sample_mask, data.n)
    # stats.adv_ratio_np = jax_math.mask_mean(
    #     jnp.logical_and(stats.advantage < 0, stats.ratio > 1), 
    #     data.sample_mask, data.n)
    # stats.adv_ratio_nn = jax_math.mask_mean(
    #     jnp.logical_and(stats.advantage < 0, stats.ratio < 1), 
    #     data.sample_mask, data.n)
    return stats


def compute_regularization(
    stats, 
    data, 
    reg_type, 
    pos_reg_coef, 
    rescaled_by_adv=False, 
    lower_threshold=-2., 
):
    if reg_type is None:
        return stats
    elif reg_type == 'simple':
        prob = lax.exp(stats.pi_logprob)
        stats.reg_below_threshold = stats.pi_logprob - data.mu_logprob < -lower_threshold
        stats.reg_above_threshold = stats.pi_logprob - data.mu_logprob > lax.log(1.2)
        stats.reg = lax.min(prob * lax.stop_gradient(
            lax.max(stats.pi_logprob - data.mu_logprob, -1.)), 
            lax.stop_gradient(prob) * lax.log(1.2))
    elif reg_type == 'wasserstein':
        stats.reg = jax_div.wasserstein(
            stats.pi_loc, stats.pi_scale, data.mu_loc, data.mu_scale)
    elif reg_type.startswith('kl'):
        kl_stats = dict(
            logp=stats.pi_logprob, 
            logq=data.mu_logprob, 
            # sample_prob=data.mu_logprob, 
            p_logits=stats.pi_logits, 
            q_logits=data.mu_logits, 
            p_loc=stats.pi_loc,  
            p_scale=stats.pi_scale, 
            q_loc=data.mu_loc, 
            q_scale=data.mu_scale, 
            logits_mask=data.action_mask, 
        )
        reg_type = reg_type.split('_')[-1]
        stats.reg = jax_div.kl_divergence(
            reg_type=reg_type, 
            **kl_stats
        )
    else:
        raise NotImplementedError(reg_type)
    stats.pos_reg = jnp.where(stats.advantage > 0, stats.reg, 0)
    if rescaled_by_adv:
        stats.pos_reg = stats.advantage * stats.pos_reg
    stats.raw_pos_reg_loss, stats.pos_reg_loss = jax_loss.to_loss(
        stats.pos_reg, 
        pos_reg_coef, 
        mask=data.sample_mask, 
        n=data.n
    )

    return stats
