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
        # To avoid eval machine precision error, strictly clip 1-action**2 to [0, 1] range
        sub = tf.reduce_sum(tf.math.log(clip_but_pass_gradient(1 - action**2, l=0, u=1) + 1e-6), axis=-1, keepdims=True)
    else:
        sub = 2 * tf.reduce_sum(tf.math.log(2.) + action - tf.nn.softplus(2 * action), axis=-1, keepdims=True)
    logpi -= sub

    return logpi
