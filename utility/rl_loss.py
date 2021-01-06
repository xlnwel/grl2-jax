import tensorflow as tf

from utility.tf_utils import assert_rank_and_shape_compatibility, static_scan


def huber_loss(x, *, y=None, threshold=1.):
    if y != None:   # if y is passed, take x-y as error, otherwise, take x as error
        x = x - y
    return tf.where(tf.abs(x) <= threshold, 
                    0.5 * tf.square(x), 
                    threshold * (tf.abs(x) - 0.5 * threshold), 
                    name='huber_loss')


def quantile_regression_loss(qtv, target, tau_hat, kappa=1., return_error=False):
    assert qtv.shape[-1] == 1, qtv.shape
    assert target.shape[-2] == 1, target.shape
    assert tau_hat.shape[-1] == 1, tau_hat.shape
    error = target - qtv           # [B, N, N']
    weight = tf.abs(tau_hat - tf.cast(error < 0, tf.float32))   # [B, N, N']
    huber = huber_loss(error, threshold=kappa)                  # [B, N, N']
    qr_loss = tf.reduce_sum(tf.reduce_mean(weight * huber, axis=2), axis=1) # [B]

    if return_error:
        return error, qr_loss
    return qr_loss


def h(x, epsilon=1e-3):
    """ h function defined in the transfomred Bellman operator 
    epsilon=1e-3 is used following recent papers(e.g., R2D2, MuZero, NGU)
    """
    sqrt_term = tf.math.sqrt(tf.math.abs(x) + 1.)
    return tf.math.sign(x) * (sqrt_term - 1.) + epsilon * x


def inverse_h(x, epsilon=1e-3):
    """ h^{-1} function defined in the transfomred Bellman operator
    epsilon=1e-3 is used following recent papers(e.g., R2D2, MuZero, NGU)
    """
    sqrt_term = tf.math.sqrt(1. + 4. * epsilon * (tf.math.abs(x) + 1. + epsilon))
    frac_term = (sqrt_term - 1.) / (2. * epsilon)
    return tf.math.sign(x) * (tf.math.square(frac_term) - 1.)


def n_step_target(reward, nth_value, discount=1., gamma=.99, steps=1., tbo=False):
    """
    discount is only the done signal
    """
    if tbo:
        return h(reward + discount * gamma**steps * inverse_h(nth_value))
    else:
        return reward + discount * gamma**steps * nth_value


def lambda_return(reward, value, discount, lambda_, bootstrap=None, axis=0):
    """
    discount includes the done signal if there is any.
    axis specifies the time dimension
    """
    if isinstance(discount, (int, float)):
        discount = discount * tf.ones_like(reward)
    # swap 'axis' with the 0-th dimension
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        discount = tf.transpose(discount, dims)
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_values = tf.concat([value[1:], bootstrap[None]], 0)
    # 1-step target: r + 𝛾 * v' * (1 - 𝝀)
    inputs = reward + discount * next_values * (1 - lambda_)
    # lambda function computes lambda return starting from the end
    target = static_scan(
        lambda acc, cur: cur[0] + cur[1] * lambda_ * acc,
        bootstrap, (inputs, discount), reverse=True
    )
    if axis != 0:
         target = tf.transpose(target, dims)
    return target


def retrace(reward, next_qs, next_action, next_pi, next_mu_a, discount, lambda_=.95, ratio_clip=1, axis=0, tbo=False):
    """
    discount = gamma * (1-done). 
    axis specifies the time dimension
    """
    if isinstance(discount, (int, float)):
        discount = discount * tf.ones_like(reward)
    if tf.rank(next_action) < tf.rank(next_pi):
        next_action = tf.one_hot(next_action, next_pi.shape[-1], dtype=next_pi.dtype)
    next_pi_a = tf.reduce_sum(next_pi * next_action, axis=-1)
    next_ratio = next_pi_a / next_mu_a
    if ratio_clip is not None:
        next_ratio = tf.minimum(next_ratio, ratio_clip)
    next_c = next_ratio * lambda_

    # swap 'axis' with the 0-th dimension
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        next_qs = tf.transpose(next_qs, dims)
        next_c = tf.transpose(next_c, dims)
        discount = tf.transpose(discount, dims)
    
    if tbo:
        next_qs = inverse_h(next_qs)
    next_v = tf.reduce_sum(next_qs * next_pi, axis=-1)
    next_q = tf.reduce_sum(next_qs * next_action, axis=-1)
    current = reward + discount * (next_v - next_c * next_q)

    assert_rank_and_shape_compatibility([current, discount, next_c], 2)
    target = static_scan(
        lambda acc, x: x[0] + x[1] * x[2] * acc,
        next_q[-1], (current, discount, next_c), 
        reverse=True)

    if axis != 0:
        target = tf.transpose(target, dims)

    if tbo:
        target = h(target)
        
    return target
