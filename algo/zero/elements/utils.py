from types import FunctionType
from typing import Tuple
import numpy as np
from jax import lax
import jax.numpy as jnp

from core.typing import AttrDict
from jax_tools import jax_assert, jax_dist, jax_loss, jax_utils


def get_hx(*args):
    hx = [a for a in args if a is not None]
    if len(hx) == 0:
        return None
    elif len(hx) == 1:
        return hx[0]
    else:
        hx = lax.concatenate(hx, -1)
    return hx

def compute_values(
    func, 
    params, 
    rng, 
    x, 
    next_x, 
    sid=None, 
    next_sid=None, 
    idx=None, 
    next_idx=None, 
    event=None, 
    next_event=None, 
    seq_axis=1
):
    hx = get_hx(sid, idx, event)
    value = func(params, rng, x, hx=hx)
    if next_x is None:
        value, next_value = jax_utils.split_data(value, axis=seq_axis)
    else:
        next_hx = get_hx(next_sid, next_idx, next_event)
        jax_assert.assert_shape_compatibility([hx, next_hx])
        next_value = func(params, rng, next_x, hx=next_hx)
    next_value = lax.stop_gradient(next_value)
    jax_assert.assert_shape_compatibility([value, next_value])

    return value, next_value

def compute_policy_dist(
    func, 
    params, 
    rng, 
    x, 
    sid, 
    idx, 
    event, 
    action_mask=None
):
    hx = get_hx(sid, idx, event)
    act_out = func(params, rng, x, hx, action_mask=action_mask)
    if isinstance(act_out, tuple):
        act_dist = jax_dist.MultivariateNormalDiag(*act_out)
    else:
        act_dist = jax_dist.Categorical(act_out)
    return act_dist


def compute_policy(
    func, 
    params, 
    rng, 
    x, 
    next_x, 
    action, 
    mu_logprob, 
    sid=None, 
    next_sid=None, 
    idx=None, 
    next_idx=None, 
    event=None, 
    next_event=None, 
    action_mask=None, 
    next_action_mask=None, 
    seq_axis=1
):
    [x, sid, idx, event, action_mask], _ = \
        jax_utils.split_data(
            [x, sid, idx, event, action_mask], 
            [next_x, next_sid, next_idx, next_event, next_action_mask], 
            axis=seq_axis
        )
    act_dist = compute_policy_dist(
        func, params, rng, x, sid, idx, event, action_mask)
    pi_logprob = act_dist.log_prob(action)
    jax_assert.assert_shape_compatibility([pi_logprob, mu_logprob])
    log_ratio = pi_logprob - mu_logprob
    ratio = lax.exp(log_ratio)

    return act_dist, pi_logprob, log_ratio, ratio

def compute_next_obs_dist(
    func, 
    params, 
    rng, 
    x, 
    action, 
):
    dist = func(params, rng, x, action)

    return dist

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

def get_basics(
    config: AttrDict, 
    env_stats: AttrDict, 
):
    if 'aid' in config:
        aid = config.aid
        n_units = len(env_stats.aid2uids[aid])
        n_envs = config.n_runners * config.n_envs
        basic_shape = (n_envs, config['n_steps'], n_units)
        shapes = env_stats['obs_shape'][aid]
        dtypes = env_stats['obs_dtype'][aid]

        action_shape = env_stats.action_shape[aid]
        action_dim = env_stats.action_dim[aid]
        action_dtype = env_stats.action_dtype[aid]
    else:
        n_envs = config.n_runners * config.n_envs
        basic_shape = (n_envs, config['n_steps'], 1)
        shapes = env_stats['obs_shape']
        dtypes = env_stats['obs_dtype']

        action_shape = env_stats.action_shape
        action_dim = env_stats.action_dim
        action_dtype = env_stats.action_dtype

    return basic_shape, shapes, dtypes, action_shape, action_dim, action_dtype

def update_data_format_with_rnn_states(
    data_format: dict, 
    config: AttrDict, 
    basic_shape: Tuple, 
    model
):
    if config.get('store_state') and config.get('rnn_type'):
        assert model.state_size is not None, model.state_size
        state_type = type(model.state_size)
        data_format['mask'] = (basic_shape, np.float32, 'mask')
        data_format['state'] = state_type(*[((None, sz), np.float32, name) 
            for name, sz in model.state_size._asdict().items()])
    
    return data_format

def get_data_format(
    config: AttrDict, 
    env_stats: AttrDict, 
    model, 
    rnn_state_fn: FunctionType=update_data_format_with_rnn_states, 
    meta=False
):
    basic_shape, shapes, dtypes, action_shape, \
        action_dim, action_dtype = \
        get_basics(config, env_stats)
    if config.timeout_done:
        obs_shape = [s+1 if i == (2 if meta else 1) else s 
            for i, s in enumerate(basic_shape)]
        data_format = {k: ((*obs_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
    else:
        obs_shape = basic_shape
        data_format = {k: ((*obs_shape, *v), dtypes[k], k) 
            for k, v in shapes.items()}
        data_format.update({f'next_{k}': ((*obs_shape, *v), dtypes[k], f'next_{k}') 
            for k, v in shapes.items()})

    data_format.update(dict(
        action=((*basic_shape, *action_shape), action_dtype, 'action'),
        value=(basic_shape, np.float32, 'value'),
        reward=(basic_shape, np.float32, 'reward'),
        discount=(basic_shape, np.float32, 'discount'),
        reset=(basic_shape, np.float32, 'reset'),
        mu_logprob=(basic_shape, np.float32, 'mu_logprob'),
    ))

    is_action_discrete = env_stats.is_action_discrete[config['aid']] \
        if isinstance(env_stats.is_action_discrete, list) else env_stats.is_action_discrete
    if is_action_discrete:
        data_format['mu'] = ((*basic_shape, action_dim), np.float32, 'mu')
    else:
        data_format['mu_mean'] = ((*basic_shape, action_dim), np.float32, 'mu_mean')
        data_format['mu_std'] = ((*basic_shape, action_dim), np.float32, 'mu_std')
    
    data_format = rnn_state_fn(
        data_format,
        config,
        basic_shape,
        model,
    )

    return data_format

def collect(buffer, env, env_step, reset, obs, next_obs, **kwargs):
    for k, v in obs.items():
        if k not in kwargs:
            kwargs[k] = v
    if buffer.config.timeout_done:
        for k, v in next_obs.items():
            kwargs[f'next_{k}'] = v
    else:
        for k, v in next_obs.items():
            kwargs[f'next_{k}'] = np.where(
                np.expand_dims(reset, -1), 
                env.prev_obs()[buffer.aid][k], 
                next_obs[k]
            )

    buffer.add(**kwargs, reset=reset)
