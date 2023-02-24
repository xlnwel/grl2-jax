from jax import lax
import jax.numpy as jnp

from core.typing import AttrDict
from tools.rms import normalize
from jax_tools import jax_assert, jax_math, jax_loss, jax_utils


def get_initial_state(state, i):
    return jax_utils.tree_map(lambda x: x[:, i], state)


def _reshape_for_bptt(*args, bptt):
    return jax_utils.tree_map(
        lambda x: x.reshape(-1, bptt, *x.shape[2:]), args
    )


def compute_values(
    func, 
    params, 
    rng, 
    x, 
    next_x, 
    state_reset, 
    state, 
    bptt, 
    seq_axis=1,
    **kwargs
):
    if state is None:
        if kwargs.get("vrnn", None) is not None:
            state_reset, next_state_reset = jax_utils.split_data(
                state_reset, axis=seq_axis)
            value, _ = func(params, rng, x, state_reset)
            next_value, _ = func(params, rng, next_x, next_state_reset)
        else:
            value, _ = func(params, rng, x)
            next_value, _ = func(params, rng, next_x) 
    else:
        state_reset, next_state_reset = jax_utils.split_data(
            state_reset, axis=seq_axis)
        if bptt is not None:
            shape = x.shape[:-1]
            x, next_x, state_reset, next_state_reset, state = \
                _reshape_for_bptt(
                    x, next_x, state_reset, next_state_reset, state, bptt=bptt
                )
        state0 = get_initial_state(state, 0)
        state1 = get_initial_state(state, 1)
        value, _ = func(params, rng, x, state_reset, state0)
        next_value, _ = func(params, rng, next_x, next_state_reset, state1)
        if bptt is not None:
            value, next_value = jax_utils.tree_map(
                lambda x: x.reshape(shape), (value, next_value)
            )
    next_value = lax.stop_gradient(next_value)
    jax_assert.assert_shape_compatibility([value, next_value])

    return value, next_value


def compute_policy_dist(
    model, 
    params, 
    rng, 
    x, 
    state_reset, 
    state, 
    action_mask=None, 
    bptt=None
):
    if state is not None and bptt is not None:
        shape = x.shape[:-1]
        x, state_reset, state, action_mask = _reshape_for_bptt(
            x, state_reset, state, action_mask, bptt=bptt
        )
    state = get_initial_state(state, 0)
    act_out, _ = model.modules.policy(
        params, rng, x, state_reset, state, action_mask=action_mask
    )
    if state is not None and bptt is not None:
        act_out = jax_utils.tree_map(
            lambda x: x.reshape(*shape, -1), act_out
        )
    act_dist = model.policy_dist(act_out)
    return act_dist


def compute_policy(
    model, 
    params, 
    rng, 
    x, 
    next_x, 
    action, 
    mu_logprob, 
    state_reset, 
    state, 
    action_mask=None, 
    next_action_mask=None, 
    bptt=None, 
    seq_axis=1
):
    [x, action_mask], _ = jax_utils.split_data(
        [x, action_mask], [next_x, next_action_mask], 
        axis=seq_axis
    )
    act_dist = compute_policy_dist(
        model, params, rng, x, state_reset, state, action_mask, bptt=bptt)
    pi_logprob = act_dist.log_prob(action)
    jax_assert.assert_shape_compatibility([pi_logprob, mu_logprob])
    log_ratio = pi_logprob - mu_logprob
    ratio = lax.exp(log_ratio)
    return act_dist, pi_logprob, log_ratio, ratio


def prefix_name(terms, name):
    if name is not None:
        new_terms = AttrDict()
        for k, v in terms.items():
            if '/' not in k:
                new_terms[f'{name}/{k}'] = v
            else:
                new_terms[k] = v
        return new_terms
    return terms


def norm_adv(
    config, 
    raw_adv, 
    sample_mask=None, 
    n=None, 
    epsilon=1e-5
):
    if config.norm_adv:
        advantage = jax_math.standard_normalization(
            raw_adv, 
            zero_center=config.get('zero_center', True), 
            mask=sample_mask, 
            n=n, 
            epsilon=epsilon, 
        )
    else:
        advantage = raw_adv
    advantage = lax.stop_gradient(advantage)
    return advantage


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
        old_value, _ = jax_utils.split_data(
            data.value, data.next_value, axis=1
        )
        raw_value_loss, stats.v_clip_frac = jax_loss.clipped_value_loss(
            stats.value, 
            v_target, 
            old_value, 
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
