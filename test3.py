import tensorflow as tf
from core.optimizer import create_optimizer
from core.tf_config import configure_gpu
from nn.func import mlp
from optimizers.rmsprop import RMSprop
from tools.timer import Timer
from tools.utils import set_seed

configure_gpu(0)
set_seed(0)

eta = tf.Variable(1, dtype=tf.float32)

def loss_fn(tape, x, **kwargs):
    x = l(x)
    loss = tf.reduce_mean(eta * (x-2)**2)
    return loss, {}


def compute_meta_grads_at_single_step(
    meta_tape, 
    eta, 
    grads, 
    vars, 
    out_grads, 
    i=None, 
):
    if i == 0:
        eta_grads = meta_tape.gradient(grads, eta, out_grads)
        return eta_grads, out_grads
    else:
        # print('out gradients', out_grads)
        with meta_tape:
            d = tf.reduce_sum([tf.reduce_sum(v * g) for v, g in zip(grads, out_grads)])
        # print('d', d)
        out_grads = [g1 + g2 for g1, g2 in zip(out_grads, meta_tape.gradient(d, vars))]
        eta_grads = meta_tape.gradient(grads, eta, out_grads)
        return eta_grads, out_grads


def compute_meta_gradients(
    meta_tape, 
    meta_loss, 
    grads_list, 
    theta, 
    eta, 
):
    inner_steps = len(grads_list)
    out_grads = meta_tape.gradient(meta_loss, theta)
    grads = []
    for i in reversed(range(inner_steps)):
        # print(i, 'outgrads', out_grads)
        new_grads, out_grads = compute_meta_grads_at_single_step(
            meta_tape, 
            eta, 
            grads_list[i], 
            theta, 
            out_grads, 
            i,
        )
        grads.append(new_grads)
    return grads

from tools.meta import *

opt_config = dict(
    opt_name=RMSprop, 
    lr=1e-3
)
l = mlp([2, 4], out_size=1, activation='relu')
meta_opt = create_optimizer([l], opt_config, f'outer_opt')
inner_opt = create_optimizer([l], opt_config, f'inner_opt')
meta_tape = tf.GradientTape(persistent=True)
# l = tf.keras.layers.Dense(1)
# eta = tf.Variable(1, dtype=tf.float32)

@tf.function
def outer(x, n):
    theta_list = [l.variables]
    grads_list = []
    for _ in range(n):
        _, tl, gl = inner_epoch(
            opt=inner_opt, 
            loss_fn=loss_fn, 
            x=x, 
            return_stats_for_meta=True
        )
        theta_list += tl
        grads_list += gl
    # print('grads list', *grads_list)
    with meta_tape:
        x = l(x)
        # print('x', x)
        loss = tf.reduce_mean((1 - x)**2)

    # print('loss', loss)
    # print(meta_tape.gradient(loss, l.variables))

    grads = compute_meta_gradients(
        meta_tape=meta_tape, 
        meta_loss=loss, 
        theta_list=theta_list, 
        grads_list=grads_list, 
        eta=eta, 
    )

    return grads


if __name__ == '__main__':
    x = tf.random.uniform((2, 3))
    print('x', x)
    l(x)
    # with Timer('mg', 1):
    #     for _ in range(1000):
    grads = outer(x, 3)
    print('grads', grads)
