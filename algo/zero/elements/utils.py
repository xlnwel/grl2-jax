from types import FunctionType
from typing import Tuple
import numpy as np
import tensorflow as tf

from core.typing import AttrDict
from tools import tf_utils


def get_hx(*args):
    hx = [a for a in args if a is not None]
    if len(hx) == 0:
        return None
    elif len(hx) == 1:
        return hx[0]
    else:
        hx = tf.concat(hx, -1)
    return hx

def compute_values(
    func, 
    x, 
    next_x, 
    sid=None, 
    next_sid=None, 
    idx=None, 
    next_idx=None, 
    event=None, 
    next_event=None
):
    hx = get_hx(sid, idx, event)
    value = func(x, hx=hx)
    if next_x is None:
        value, next_value = tf_utils.split_data(value)
    else:
        next_hx = get_hx(next_sid, next_idx, next_event)
        next_value = func(next_x, hx=next_hx)
    next_value = tf.stop_gradient(next_value)

    return value, next_value

def compute_policy(
    func, 
    obs, 
    next_obs, 
    action, 
    mu_logprob, 
    sid=None, 
    next_sid=None, 
    idx=None, 
    next_idx=None, 
    event=None, 
    next_event=None, 
    action_mask=None
):
    [obs, sid, idx, event], _ = tf_utils.split_data(
        [obs, sid, idx, event], 
        [next_obs, next_sid, next_idx, next_event]
    )
    hx = get_hx(sid, idx, event)
    act_dist = func(obs, hx=hx, action_mask=action_mask)
    pi_logprob = act_dist.log_prob(action)
    tf_utils.assert_rank_and_shape_compatibility([pi_logprob, mu_logprob])
    log_ratio = pi_logprob - mu_logprob
    ratio = tf.exp(log_ratio)

    return act_dist, pi_logprob, log_ratio, ratio

def prefix_name(terms, name):
    if name is not None:
        new_terms = {}
        for k, v in terms.items():
            if '/' not in k:
                new_terms[f'{name}/{k}'] = v
            else:
                new_terms[k] = v
        return new_terms
    return terms

def compute_inner_steps(config):
    if config.K is not None and config.L is not None:
        config.inner_steps = config.K + config.L
    else:
        config.inner_steps = None
    if config.inner_steps == 0:
        config.inner_steps = None

    return config

def get_rl_module_names(model):
    keys = [k for k in model.keys() if not k.startswith('meta')]
    return keys

def get_meta_module_names(model):
    keys = [k for k in model.keys() if k.startswith('meta') and k != 'meta']
    return keys

def get_meta_param_module_names(model):
    keys = [k for k in model.keys() if k == 'meta']
    return keys

def get_rl_modules(model):
    keys = get_rl_module_names(model)
    modules = tuple([model[k] for k in keys])
    return modules

def get_meta_modules(model):
    keys = get_meta_module_names(model)
    modules = tuple([model[k] for k in keys])
    return modules

def get_meta_param_modules(model):
    keys = get_meta_param_module_names(model)
    modules = tuple([model[k] for k in keys])
    return modules

def get_basics(
    config: AttrDict, 
    env_stats: AttrDict, 
    model
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
        dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        state_type = type(model.state_size)
        data_format['mask'] = (basic_shape, tf.float32, 'mask')
        data_format['state'] = state_type(*[((None, sz), dtype, name) 
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
        get_basics(config, env_stats, model)
    if meta:
        basic_shape = (config.inner_steps + config.extra_meta_step,) + basic_shape
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
        value=(basic_shape, tf.float32, 'value'),
        reward=(basic_shape, tf.float32, 'reward'),
        discount=(basic_shape, tf.float32, 'discount'),
        reset=(basic_shape, tf.float32, 'reset'),
        mu_logprob=(basic_shape, tf.float32, 'mu_logprob'),
    ))

    is_action_discrete = env_stats.is_action_discrete[config['aid']] if isinstance(env_stats.is_action_discrete, list) else env_stats.is_action_discrete
    if is_action_discrete:
        data_format['mu'] = ((*basic_shape, action_dim), tf.float32, 'mu')
    else:
        data_format['mu_mean'] = ((*basic_shape, action_dim), tf.float32, 'mu_mean')
        data_format['mu_std'] = ((*basic_shape, action_dim), tf.float32, 'mu_std')
    
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
