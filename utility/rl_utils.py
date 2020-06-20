import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy

from utility.tf_utils import static_scan


def huber_loss(x, *, y=None, threshold=1.):
    if y != None:   # if y is passed, take x-y as error, otherwise, take x as error
        x = x - y
    return tf.where(tf.abs(x) <= threshold, 
                    0.5 * tf.square(x), 
                    threshold * (tf.abs(x) - 0.5 * threshold), 
                    name='huber_loss')

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
        sub = tf.reduce_sum(
            tf.math.log(clip_but_pass_gradient(1 - action**2, l=0, u=1) + 1e-8), 
            axis=-1)
    else:
        sub = 2 * tf.reduce_sum(
            tf.cast(tf.math.log(2.), action.dtype) 
            - action - tf.nn.softplus(-2 * action), 
            axis=-1)
    assert logpi.shape.ndims == sub.shape.ndims, f'{logpi.shape} vs {sub.shape}'
    logpi -= sub

    return logpi

def h(x, epsilon=1e-3):
    """ h function defined in the transfomred Bellman operator """
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
    returns = static_scan(
        lambda acc, cur: cur[0] + cur[1] * lambda_ * acc,
        bootstrap, (inputs, discount), reverse=True
    )
    if axis != 0:
         returns = tf.transpose(returns, dims)
    return returns

def retrace_lambda(reward, q, next_value, next_ratio, discount, lambda_=.95, ratio_clip=1, axis=0, tbo=False):
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
        q = tf.transpose(q, dims)
        next_value = tf.transpose(next_value, dims)
        next_ratio = tf.transpose(next_ratio, dims)
        discount = tf.transpose(discount, dims)

    if ratio_clip is not None:
        next_ratio = tf.minimum(next_ratio, ratio_clip)
    next_ratio *= lambda_
    
    if tbo:
        q = inverse_h(q)
        next_value = inverse_h(next_value)
    delta = reward + discount * next_value - q

    # because we generally assume q_T - Q_T == 0, we do not need 𝜌_T
    assert delta.shape[0] == next_ratio.shape[0] + 1, f'{delta.shape} vs {next_ratio.shape}'
    # we starts from delta[-1] as we generally assume q_T - Q_T == 0,
    diff = static_scan(
        lambda acc, x: x[0] + x[1] * x[2] * acc,
        delta[-1], (delta[:-1], discount[:-1], next_ratio), 
        reverse=True)
    diff = tf.concat([diff, delta[-1:]], axis=0)
    returns = q + diff

    if axis != 0:
        returns = tf.transpose(returns, dims)

    if tbo:
        returns = h(returns)
        
    return returns

def apex_epsilon_greedy(env_id, n_envs, epsilon=.4, alpha=8):
    # the 𝝐-greedy schedule used in Ape-X and Agent57
    if n_envs == 1:
        return epsilon
    else:
        return epsilon ** (1 + env_id / (n_envs - 1) * alpha)
