import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from core.optimizer import build_optimizer, optimize


n_steps = 2
def mlp_fn(x):
    l = hk.Sequential([hk.Linear(1), jax.nn.relu, hk.Linear(1)])
    return l(x)
net = hk.transform(mlp_fn)
rng = jax.random.PRNGKey(42)


def inner_loss(theta, eta, x):
    x = net.apply(theta, None, x)
    # print('inner x', x)
    loss = jnp.mean(eta * (x-2)**2)
    return loss, {}

def outer_loss(eta, theta, opt_state, x, opt):
    for _ in range(n_steps):
        theta, opt_state, _ = optimize(
            inner_loss, 
            params=theta, 
            state=opt_state, 
            kwargs=dict(
                eta=eta, 
                x=x, 
            ), 
            opt=opt, 
            name='theta'
        )
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
    config = dict(
        opt_name='adam', 
        lr=3e-4, 
        clip_norm=.5,
        b1=0, 
        b2=0, 
        # eps_root=1e-4, 
        eps=1
    )
    np.random.seed(42)
    eta = np.array(1, dtype='float32')
    x = np.random.uniform(size=(2, 3))
    theta = net.init(rng, x)
    print('theta', theta)
    theta_opt, theta_state = build_optimizer(
        params=theta, 
        **config, 
        name='theta'
    )
    eta_opt, eta_state = build_optimizer(
        params=eta, 
        **config, 
        name='eta'
    )
    print('meta state', eta_state)
    theta, opt_state, eta, eta_opt_state, loss = blo(
        theta, eta, theta_state, eta_state, x, theta_opt, eta_opt)
    # print(eta)
