import numpy as np
import jax
from jax import lax, nn, random
import jax.numpy as jnp

from jax_tools import jax_assert, jax_loss


def get_initial_state(state, i):
    return jax.tree_util.tree_map(lambda x: x[:, i], state)


def _reshape_for_bptt(*args, bptt):
    return jax.tree_util.tree_map(
        lambda x: x.reshape(-1, bptt, *x.shape[2:]), args
    )


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).astype(jnp.float32)
    clip_low = (x < l).astype(jnp.float32)
    return x + lax.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


def concate_along_unit_dim(x):
    x = jnp.concatenate(x, axis=1)
    return x


def logprob_correction(action, logprob, is_action_squashed):
    """ 
    This function is used to correct logpi from a Gaussian distribution 
    when sampled action is squashed by tanh into [0, 1] range 
    is_action_squashed indicate if action has been squashed
    """
    if is_action_squashed:
        # To avoid evil machine precision error, strictly clip 1-action**2 to [0, 1] range
        sub = jnp.sum(lax.log(clip_but_pass_gradient(
            1 - action**2, l=0, u=1) + 1e-8), axis=-1)
    else:
        sub = 2 * jnp.sum(lax.log(2.) - action - nn.softplus(-2 * action), axis=-1)
    assert logprob.ndim == sub.ndim, f'{logprob.shape} vs {sub.shape}'
    logprob = logprob - sub

    return logprob


def joint_actions(actions):
    assert actions.ndim == 4, actions.shape
    all_actions = [actions]
    # roll along the unit dimension
    for _ in range(1, actions.shape[-2]):
        actions = jnp.roll(actions, 1, -2)
        all_actions.append(actions)
    all_actions = jnp.concatenate(all_actions, -1)

    return all_actions


def concat_sa(x, a):
    x = jnp.concatenate([x, a], -1)

    return x


def compute_action_logprob(
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
    rngs = random.split(rng)
    state = get_initial_state(state, 0)
    act_out, _ = model.modules.policy(
        params, rngs[0], x, state_reset, state, 
        action_mask=action_mask
    )
    if state is not None and bptt is not None:
        act_out = jax.tree_util.tree_map(
            lambda x: x.reshape(*shape, -1), act_out
        )
    act_dist = model.policy_dist(act_out)
    raw_action, raw_logprob = act_dist.sample_and_log_prob(
        seed=rngs[1], joint=True)
    if model.is_action_discrete:
        action, logprob = raw_action, raw_logprob
    else:
        action = jnp.tanh(raw_action)
        logprob = logprob_correction(
            raw_action, raw_logprob, is_action_squashed=False)
    
    return action, logprob, act_dist


def compute_joint_action_logprob(
    model, 
    params, 
    rng, 
    data, 
    bptt=None
):
    action = []
    logprob = []
    act_dists = []
    for p, uids in zip(params, model.aid2uids):
        d = data.slice((slice(None), slice(None), uids))
        policy_state = None if d.state is None else d.state.policy
        a, lp, d = compute_action_logprob(
            model, 
            p, 
            rng, 
            d.obs, 
            d.state_reset, 
            policy_state, 
            action_mask=d.action_mask, 
            bptt=bptt
        )
        action.append(a)
        logprob.append(lp)
    action = jnp.concatenate(action, 2)
    action = action.reshape(*action.shape[:2], 1, -1)
    logprob = jnp.sum(jnp.concatenate(logprob, 2), 
        2, keepdims=True)
    act_dists.append(d)

    return action, logprob, act_dists


def compute_qs(
    func, 
    params, 
    rng, 
    x, 
    action, 
    state_reset, 
    states, 
    bptt=None, 
    return_minimum=False
):
    qs = []
    rngs = random.split(rng, len(params))
    if states is None:
        states = [None for _ in params]
    else:
        if bptt is not None:
            shape = x.shape[:-1]
            x, state_reset, state = _reshape_for_bptt(
                x, state_reset, state, bptt=bptt
            )
    for p, rng, state in zip(params, rngs, states):
        if state is None:
            q, _ = func(p, rng, x, action)
        else:
            state = get_initial_state(state, 0)
            q, _ = func(p, rng, x, action, state_reset, state)
            if bptt is not None:
                q = q.reshape(shape)
        qs.append(q)
    
    if return_minimum:
        return jnp.min(jnp.stack(qs), 0)
    else:
        return qs


def compute_target(
    reward, discount, gamma, next_q, temp, next_logprob
):
    jax_assert.assert_shape_compatibility([
        reward, discount, next_q, next_logprob
    ])
    next_value = next_q - temp * next_logprob
    return reward + discount * gamma * next_value


def compute_q_loss(
    config, qs, q_target, data, stats
):
    if config.get('value_sample_mask', False):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    reps = (len(qs), 1, 1, 1)
    qs = jnp.stack(qs)
    assert q_target.ndim == 3, q_target.ndim
    q_target = jnp.tile(q_target, reps)
    jax_assert.assert_shape_compatibility([qs, q_target])
    raw_qs_loss = .5 * (qs - q_target)**2
    stats.raw_qs_loss = raw_qs_loss
    raw_q_loss = jnp.sum(raw_qs_loss, 0)
    stats.scaled_q_loss, q_loss = jax_loss.to_loss(
        raw_q_loss, 
        coef=stats.q_coef, 
        mask=sample_mask, 
        n=data.n
    )
    stats.q_loss = q_loss

    return q_loss, stats


def compute_policy_loss(
    config, q, logprob, temp, data, stats
):
    if not config.get('policy_sample_mask', True):
        sample_mask = data.sample_mask
    else:
        sample_mask = None

    raw_loss = temp * logprob - q
    scaled_loss, loss = jax_loss.to_loss(
        raw_loss, 
        coef=stats.policy_coef, 
        mask=sample_mask, 
        n=data.n
    )
    stats.raw_policy_loss = raw_loss
    stats.scaled_policy_loss = scaled_loss
    stats.policy_loss = loss

    return loss, stats
