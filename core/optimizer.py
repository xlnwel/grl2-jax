import collections
import logging
from typing import Dict, Sequence, Union
import jax
import jax.numpy as jnp
import optax

from core.log import do_logging
from core.typing import AttrDict
from tools.utils import add_prefix, flatten_dict, is_empty
from jax_tools.jax_utils import compute_norms


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
    NamedTuple = collections.namedtuple(name, [
        a.init.__qualname__.split('.', 1)[0] for a in args])
    def init_fn(params):
        return NamedTuple(*[fn(params) for fn in init_fns])

    def update_fn(updates, state, params=None):
        assert updates is not None, name
        assert state is not None, name
        assert len(update_fns) == len(state), (len(update_fns), len(state))

        updates_list = []
        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params)
            updates_list.append(updates)
            new_state.append(new_s)
        return NamedTuple(*updates_list), NamedTuple(*new_state)

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
    opts.append(select_optimizer(opt_name)(lr, **opt_kwargs))
    opt = chain(*opts, name=name)

    if params is not None:
        state = opt.init(params)
        return opt, state
    return opt


def compute_gradients(
    loss_fn, 
    params: Dict, 
    kwargs: Dict, 
    debug=False, 
    name=None
):
    grads, stats = jax.grad(
        loss_fn, has_aux=True)(params, **kwargs)
    stats = _record_grads(stats, grads, name=name, debug=debug)
    return grads, stats


def compute_meta_gradients(
    loss_fn, 
    params: Dict, 
    kwargs: Dict, 
    name=None, 
    debug=False, 
):
    grads, (state, stats) = jax.grad(
        loss_fn, has_aux=True)(params, **kwargs)
    stats = _record_grads(stats, grads, name=name, debug=debug)
    return grads, (state, stats)


def compute_updates(
    grads: Dict, 
    state, 
    params, 
    opt, 
    stats, 
    name=None, 
    debug=False, 
):
    updates, state = opt.update(grads, state, params)
    for k, v in updates._asdict().items():
        k = add_prefix(k, name)
        if debug:
            if is_empty(v):
                continue
            updates_norm = compute_norms(v)
            updates_norm = flatten_dict(
                updates_norm, prefix=k, suffix='updates/norm')
            stats.update(updates_norm)
        global_norm = optax.global_norm(v)
        stats[f'{k}/total_updates/norm'] = global_norm

    return updates[-1], state, stats


def apply_updates(params, updates):
    params = optax.apply_updates(params, updates)
    return params


def optimize(
    loss_fn, 
    params: Dict, 
    state: Union[Dict, Sequence], 
    kwargs: Dict, 
    opt, 
    name=None, 
    debug=False, 
):
    grads, stats = compute_gradients(
        loss_fn=loss_fn, params=params, 
        kwargs=kwargs, name=name, 
        debug=debug)
    updates, state, stats = compute_updates(
        grads, state, params, opt, stats, name=name, debug=debug)
    params = apply_updates(params, updates)
    return params, state, stats


def _record_grads(stats, grads, name, debug):
    if debug:
        grads_norm = compute_norms(grads)
        grads_norm = flatten_dict(
            grads_norm, prefix=name, suffix='grads/norm')
        stats.update(grads_norm)
    global_norm = optax.global_norm(grads)
    k = add_prefix('total_grads/norm', name)
    stats[k] = global_norm

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
    opt, state = build_optimizer(params=theta, opt_name='adam', lr=1)
    params, state, stats = optimize(loss, theta, state, **data)
    print(params)
    params, state, stats = optimize(loss, theta, state, **data)
    print(params)
