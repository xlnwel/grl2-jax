from jax import lax
import jax.numpy as jnp

from algo.dynamics.elements.utils import compute_mean_logvar, joint_actions, combine_sa
from algo.ppo.elements.utils import get_initial_state, _reshape_for_bptt, \
    compute_values, prefix_name, norm_adv, compute_vf_loss, record_policy_stats, record_target_adv, \
    compute_policy, compute_policy_dist
from jax_tools import jax_assert, jax_utils, jax_math, jax_loss


def compute_actor_loss(
    config, 
    data, 
    stats, 
    act_dists, 
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
    if raw_pg_loss.ndim == 4:   # reduce the action dimension for continuous actions
        raw_pg_loss = jnp.sum(raw_pg_loss, axis=-1)
    scaled_pg_loss, pg_loss = jax_loss.to_loss(
        raw_pg_loss, 
        stats.pg_coef, 
        mask=sample_mask, 
        n=data.n
    )
    stats.raw_pg_loss = raw_pg_loss
    stats.scaled_pg_loss = scaled_pg_loss
    stats.pg_loss = pg_loss

    print('--->>>> n'*20)
    assert 0
    for act_dist in act_dists:
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