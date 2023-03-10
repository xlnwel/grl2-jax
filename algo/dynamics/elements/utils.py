from jax import nn
import jax.numpy as jnp

from core.typing import AttrDict


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
