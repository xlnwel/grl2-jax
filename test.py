import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from nn.func import mlp

n_steps = 2
def mlp_fn(x):
    l = mlp([2, 4], out_size=1, activation='relu')
    return l(x)
net = hk.transform(mlp_fn)
rng = jax.random.PRNGKey(42)


def inner_loss(theta, eta, x):
    x = net.apply(theta, None, x)
    # print('inner x', x)
    loss = jnp.mean(eta * (x-2)**2)
    return loss

def inner_update(theta, eta, opt_state, x, opt):
    loss, g = jax.value_and_grad(inner_loss)(theta, eta, x)
    updates, opt_state = opt.update(g, opt_state)
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state, loss

def outer_loss(eta, theta, opt_state, x, opt):
    for _ in range(n_steps):
        theta, opt_state, loss = inner_update(theta, eta, opt_state, x, opt)
    x = net.apply(theta, None, x)
    ol = jnp.mean((1-x)**2)
    return ol, (theta, opt_state)

def blo(theta, eta, opt_state, eta_opt_state, x, opt, eta_opt):
    (loss, (theta, opt_state)), mg = jax.value_and_grad(
        outer_loss, has_aux=True)(eta, theta, opt_state, x, opt)
    update, eta_opt_state = eta_opt.update(mg, eta_opt_state)
    print('eta state', eta_opt_state)
    eta = optax.apply_updates(eta, update)
    return theta, opt_state, eta, eta_opt_state, loss


if __name__ == '__main__':
    np.random.seed(42)
    eta = np.array(1, dtype='float32')
    x = np.random.uniform(size=(2, 3))
    theta = net.init(rng, x)
    print('theta', theta)
    meta_opt = optax.rmsprop(1)
    meta_opt = optax.chain(optax.clip_by_global_norm(.5), meta_opt)
    meta_state = meta_opt.init(eta)
    print('meta state', meta_state)
    inner_opt = getattr(optax, 'adam')(1)
    # inner_opt = optax.chain(optax.clip_by_global_norm(.5), meta_opt)
    inner_state = inner_opt.init(theta)
    import cloudpickle
    with open('test.pkl', 'wb') as f:
        cloudpickle.dump(inner_state, f)
    theta, opt_state, eta, eta_opt_state, loss = blo(theta, eta, inner_state, meta_state, x, inner_opt, meta_opt)
    # print(eta)
