import tensorflow as tf


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


def logpi_correction(action, logpi, is_action_squashed):
    """ 
    This function is used to correct logpi from a Gaussian distribution 
    when sampled action is squashed by tanh into [0, 1] range 
    is_action_squashed indicate if action has been squashed
    """
    if is_action_squashed:
        # To avoid evil machine precision error, strictly clip 1-action**2 to [0, 1] range
        sub = tf.reduce_sum(tf.math.log(clip_but_pass_gradient(1 - action**2, l=0, u=1) + 1e-8), axis=-1)
    else:
        sub = 2 * tf.reduce_sum(tf.math.log(2.) - action - tf.nn.softplus(-2 * action), axis=-1)
    assert len(logpi.shape) == len(sub.shape) == 1, f'{logpi.shape} vs {sub.shape}'
    logpi -= sub

    return logpi

def n_step_target(reward, done, nth_value, gamma, steps=1):
    with tf.name_scope('n_step_target'):
        return tf.stop_gradient(reward + gamma**steps * (1. - done) * nth_value)

def h(x, epsilon=1e-2):
    """h function defined in Ape-X DQfD"""
    sqrt_term = tf.math.sqrt(tf.math.abs(x) + 1)
    return tf.math.sign(x) * (sqrt_term - 1) + epsilon * x

def inverse_h(x, epsilon=1e-2):
    """h^{-1} function defined in Ape-X DQfD"""
    sqrt_term = tf.math.sqrt(1 + 4 * epsilon * (tf.math.abs(x) + 1 + epsilon))
    frac_term = (sqrt_term - 1) / (2 * epsilon)
    return tf.math.sign(x) * (frac_term ** 2 - 1)

def transformed_n_step_target(reward, done, nth_value, gamma, steps):
    """Transformed Bellman operator defined in Ape-X DQfD"""
    with tf.name_scope('n_step_target'):
        return tf.stop_gradient(h(reward + gamma**steps * (1. - done) * inverse_h(nth_value)))
