from jax import nn
import jax.numpy as jnp

import collections
from core.typing import AttrDict
from typing import List


NormalizerParams = collections.namedtuple('NormalizerParams', ('mean', 'std', 'n', 'eps'))

def construct_normalizer_params(shape, eps=1e-8):
    return NormalizerParams(
        mean=jnp.zeros(shape),
        std=jnp.ones(shape),
        n=jnp.zeros([]),
        eps=eps
    )
    
def normalizer_update(params, samples):
    old_mean, old_std, old_n = params.mean, params.std, params.n
    samples = samples - old_mean
    
    m = samples.shape[0]
    delta = samples.mean(axis=0)
    new_n = old_n + m
    new_mean = old_mean + delta * m / new_n
    new_std = jnp.sqrt((old_std**2 * old_n + samples.var(axis=0) * m + delta**2 * old_n * m / new_n) / new_n)
    params = NormalizerParams(
        mean=new_mean,
        std=new_std,
        n=new_n,
        eps=params.eps
    )
    return params

class SingleNormalizer:
    def __init__(self, name: str, shape: List[int]):  # batch_size x ...
        self.name = name
        self.shape = shape
        
    def normalize(self, params, x, inverse=False):
        if inverse:
            return x * params.std + params.mean
        return (x - params.mean) / params.std.clip(min=params.eps)

class Normalizers:
    def __init__(self, dim_obs: int):
        self.obs = SingleNormalizer('obs', [dim_obs])
        self.diff = SingleNormalizer('diff', [dim_obs])


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


def compute_mean_logvar(x, max_logvar, min_logvar):
    mean, logvar = jnp.split(x, 2, axis=-1)
    logvar = max_logvar - nn.softplus(max_logvar - logvar)
    logvar = min_logvar + nn.softplus(logvar - min_logvar)

    return mean, logvar


def joint_actions(actions):
    all_actions = [actions]
    # roll along the unit dimension
    for _ in range(1, actions.shape[-2]):
        actions = jnp.roll(actions, 1, -2)
        all_actions.append(actions)
    all_actions = jnp.concatenate(all_actions, -1)

    return all_actions


def combine_sa(x, a):
    a = joint_actions(a)
    x = jnp.concatenate([x, a], -1)

    return x
