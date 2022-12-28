from jax import lax, random
import jax.numpy as jnp

from core.typing import dict2AttrDict
from jax_tools import jax_loss, jax_math, jax_utils
from tools.utils import prefix_name
from .utils import compute_values, compute_policy
from algo.zero.elements.loss import Loss as LossBase, \
    compute_actor_loss, compute_vf_loss, record_target_adv, record_policy_stats


class Loss(LossBase):
    def img_loss(
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
            bptt=self.config.img_vrnn_bptt, 
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
                data.state_reset[:, :-1], 
                None if data.state is None else data.state.policy, 
                action_mask=data.action_mask, 
                next_action_mask=data.next_action_mask, 
                bptt=self.config.img_prnn_bptt, 
                seq_axis=1, 
            )
        stats = record_policy_stats(data, stats, act_dist)

        v_target, stats.raw_adv = jax_loss.compute_target_advantage(
            config=self.config, 
            reward=data.reward, 
            discount=data.discount, 
            reset=data.reset, 
            value=lax.stop_gradient(stats.value), 
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
        kl_stats = jax_utils.tree_map(lambda x: jnp.split(x, 2)[1], kl_stats)
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

        stats = prefix_name(stats, name)

        return loss, stats


def create_loss(config, model, name='zero'):
    loss = Loss(config=config, model=model, name=name)

    return loss
