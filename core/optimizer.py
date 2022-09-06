import collections
import logging
from typing import Dict, List, Tuple, Union
import jax
import jax.numpy as jnp
import optax

from core.log import do_logging
from core.typing import AttrDict
from tools.utils import is_empty


logger = logging.getLogger(__name__)

def select_optimizer(name):
    # add custom optimizers here
    if isinstance(name, str):
        return getattr(optax, name.lower())
    return name


def chain(
    *args: optax.GradientTransformation,
    name: str
) -> optax.GradientTransformation:
    """Applies a list of chainable update transformations.

    Given a sequence of chainable transforms, `chain` returns an `init_fn`
    that constructs a `state` by concatenating the states of the individual
    transforms, and returns an `update_fn` which chains the update transformations
    feeding the appropriate state to each.

    Args:
        *args: a sequence of chainable (init_fn, update_fn) tuples.
        name: name for namedtuple
    Returns:
        A single (names, (init_fn, update_fn)) tuple.
    """

    init_fns, update_fns = zip(*args)
    NT = collections.namedtuple(name, [
        a.init.__qualname__.split('.', 1)[0] for a in args])
    def init_fn(params):
        return NT(*[fn(params) for fn in init_fns])

    def update_fn(updates, state, params=None):
        if len(update_fns) != len(state):
            raise ValueError('The number of updates and states has to be the same in '
                'chain! Make sure you have called init first!')

        updates_list = []
        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params)
            updates_list.append(updates)
            new_state.append(new_s)
        return NT(*updates_list), NT(*new_state)

    return optax.GradientTransformation(init_fn, update_fn)


def build_optimizer(
    *,
    params=None, 
    opt_name='adam', 
    lr, 
    clip_norm: float=None, 
    weight_decay: float=None, 
    name: str, 
    **opt_kwargs, 
):
    opts = []
    if weight_decay:
        opts.append(optax.add_decayed_weights(weight_decay))
    if clip_norm:
        opts.append(optax.clip_by_global_norm(clip_norm))
    opt = chain(
        *opts, 
        select_optimizer(opt_name)(lr, **opt_kwargs), 
        name=name
    )

    if params is not None:
        state = opt.init(params)
        return opt, state
    return opt

def compute_gradients(
    loss_fn, 
    params: Dict, 
    kwargs: Dict, 
    name: str, 
    by_part=False, 
):
    grads, stats = jax.grad(loss_fn, has_aux=True)(params, **kwargs)
    stats = _record_grads(stats, grads, name, by_part)
    return grads, stats

def compute_meta_gradients(
    loss_fn, 
    params: Dict, 
    kwargs: Dict, 
    name: str, 
    by_part=False
):
    grads, (stats, inner_opt_state) = jax.grad(
        loss_fn, has_aux=True)(params, **kwargs)
    stats = _record_grads(stats, grads, name, by_part)
    return grads, (stats, inner_opt_state)

def compute_updates(
    grads: Dict, 
    state, 
    opt, 
    stats, 
    name: str
):
    updates, state = opt.update(grads, state)
    for k, v in updates._asdict().items():
        v, _ = jax.tree_util.tree_flatten(v)
        if is_empty(v):
            continue
        v = jnp.stack(jax.tree_map(jnp.linalg.norm, v))
        stats[f'{name}/{k}/updates_norm'] = v
        stats[f'{name}/{k}/total_updates_norm'] = jnp.sum(v)

    return updates[-1], state, stats

def apply_updates(params, updates):
    params = optax.apply_updates(params, updates)

    return params

def optimize(
    loss_fn, 
    params: Dict, 
    state: Union[Dict, Tuple, List], 
    kwargs: Dict, 
    opt, 
    name
):
    grads, stats = compute_gradients(
        loss_fn=loss_fn, params=params, kwargs=kwargs, name=name)
    updates, state, stats = compute_updates(
        grads, state, opt, stats, name)
    params = apply_updates(params, updates)
    return params, state, stats

def _record_grads(stats, grads, name, by_part):
    if by_part:
        for k, v in grads.items():
            v, _ = jax.tree_util.tree_flatten(v)
            if v:
                grads_norm = jnp.stack(jax.tree_map(jnp.linalg.norm, v))
                stats[f'{name}/{k}/grads_norm'] = grads_norm
                stats[f'{name}/{k}/total_grads_norm'] = jnp.sum(grads_norm)
    else:
        raw_grads, _ = jax.tree_util.tree_flatten(grads)
        if raw_grads:
            grads_norm = jnp.stack(jax.tree_map(jnp.linalg.norm, raw_grads))
            stats[f'{name}/grads_norm'] = grads_norm
            stats[f'{name}/total_grads_norm'] = jnp.sum(grads_norm)
    return stats


if __name__ == '__main__':
    import numpy as np
    import haiku as hk
    import jax.numpy as jnp
    eta = np.array(1, dtype='float32')
    x = np.random.uniform(size=(2, 3))
    data = AttrDict()
    data.x = x
    def layer(data):
        l = hk.Linear(2)
        return l(data['x'])
    net = hk.transform(layer)
    rng = jax.random.PRNGKey(42)
    theta = net.init(rng, data)
    def loss(params, data):
        x = net.apply(params, None, data)
        l = jnp.mean((1. - x)**2)
        return l, {}
    opt, state = create_optimizer(params=theta, opt_name='adam', lr=1)
    params, state, stats = optimize(loss, theta, state, **data)
    print(params)
    params, state, stats = optimize(loss, theta, state, **data)
    print(params)