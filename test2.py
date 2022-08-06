import tensorflow as tf
from core.tf_config import configure_gpu
from nn.func import mlp
from optimizers.rmsprop import RMSprop
from utility.utils import set_seed

configure_gpu(0)
set_seed(0)

meta_opt = RMSprop(1e-3)
inner_opt = RMSprop(1e-3)
n_steps = 2
meta_tape = tf.GradientTape(persistent=True)
l = mlp([2, 4], out_size=1, activation='relu')
# l = tf.keras.layers.Dense(1)
c = tf.Variable(1,dtype=tf.float32)


def compute_hessian_vector(tape, vars1, vars2, out_grads):
    """ Compute the gradient of vars1
    vars2 must be a differentiable function of vars1
    """
    # with tape:
    #     out = tf.reduce_sum([tf.reduce_sum(v * g) for v, g in zip(vars2, out_grads)])
    new_grads = tape.gradient(vars2, vars1, out_grads)
    return new_grads

def compute_grads_through_optmizer(tape, vars, grads, trans_grads, out_grads):
    out_grads = compute_hessian_vector(
        tape, 
        grads, 
        trans_grads, 
        out_grads
    )
    grads = compute_hessian_vector(
        tape, 
        vars, 
        grads, 
        out_grads
    )
    return grads

# @tf.function
def inner(x):
    with meta_tape:
        with tf.GradientTape() as inner_tape:
            x = l(x)
            # print('inner x', x)
            loss = tf.reduce_mean(c * (x-2)**2)
        grads = inner_tape.gradient(loss, l.variables)
        inner_opt.apply_gradients(zip(grads, l.variables))
    trans_grads = list(inner_opt.get_transformed_grads().values())
    return trans_grads


# @tf.function
def meta(
    x, 
    vars1, 
    vars2, 
    trans_grads1, 
    trans_grads2, 
    trans_grads3, 
):
    with meta_tape:
        x = l(x)
        loss = tf.reduce_mean((1 - x)**2)
    out_grads = meta_tape.gradient(loss, l.variables)
    grads3 = meta_tape.gradient(
        trans_grads3, c, out_grads
    )
    out_grads1 = meta_tape.gradient(
        trans_grads3, vars2, out_grads
    )
    out_grads = [o1 + o2 for o1, o2 in zip(out_grads, out_grads1)]
    
    grads2 = meta_tape.gradient(
        trans_grads2, c, out_grads
    )
    out_grads1 = meta_tape.gradient(
        trans_grads2, vars1, out_grads
    )
    out_grads = [o1 + o2 for o1, o2 in zip(out_grads, out_grads1)]
    grads1 = meta_tape.gradient(
        trans_grads1, c, out_grads
    )
    grads_list = [
        grads3, grads2, grads1]
    return grads_list


# @tf.function
def outer(x):
    trans_grads1 = inner(x)
    var1 = l.variables
    trans_grads2 = inner(x)
    var2 = l.variables
    trans_grads3 = inner(x)
    # print('grads list', trans_grads1, trans_grads2, trans_grads3)
    grads = meta(
        x, 
        var1,
        var2,
        trans_grads1,
        trans_grads2, 
        trans_grads3, 
    )
    return grads

if __name__ == '__main__':
    x = tf.random.uniform((2, 3))
    print('x', x)
    grads = outer(x)
    print('grads', grads)