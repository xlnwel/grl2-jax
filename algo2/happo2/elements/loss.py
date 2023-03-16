from jax import lax, random
import jax.numpy as jnp

from core.elements.loss import LossBase
from core.typing import dict2AttrDict
from jax_tools import jax_loss
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


def create_loss(config, model, name='happo'):
    loss = Loss(config=config, model=model, name=name)

    return loss
