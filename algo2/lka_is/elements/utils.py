from types import FunctionType
from typing import Tuple
import numpy as np
from jax import lax
import jax.numpy as jnp

from core.typing import AttrDict
from jax_tools import jax_assert, jax_dist, jax_loss, jax_utils


def _get_initial_state(state, i):
    return jax.tree_util.tree_map(lambda x: x[:, i], state)

def _reshape_for_bptt(*args, bptt):
    return jax.tree_util.tree_map(
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
    seq_axis=1
):
    if state is None:
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
        state0 = _get_initial_state(state, 0)
        state1 = _get_initial_state(state, 1)
        value, _ = func(params, rng, x, state_reset, state0)
        next_value, _ = func(params, rng, next_x, next_state_reset, state1)
        if bptt is not None:
            value, next_value = jax.tree_util.tree_map(
                lambda x: x.reshape(shape), (value, next_value)
            )
    next_value = lax.stop_gradient(next_value)
    jax_assert.assert_shape_compatibility([value, next_value])

    return value, next_value

def compute_policy_dist(
    func, 
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
    state = _get_initial_state(state, 0)
    act_out, _ = func(
        params, rng, x, state_reset, state, action_mask=action_mask
    )
    if state is not None and bptt is not None:
        act_out = jax.tree_util.tree_map(
            lambda x: x.reshape(*shape, -1), act_out
        )
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
        func, params, rng, x, state_reset, state, action_mask, bptt=bptt)
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
    for k, v in next_obs.items():
        kwargs[f'next_{k}'] = v
    buffer.add(**kwargs, reset=reset)
