import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility.tf_utils import assert_rank, assert_shape_compatibility


def huber_loss(x, *, y=None, threshold=1.):
    if y != None:   # if y is passed, take x-y as error, otherwise, take x as error
        assert_shape_compatibility([x, y])
        x = x - y
    return tf.where(tf.abs(x) <= threshold, 
                    0.5 * tf.square(x), 
                    threshold * (tf.abs(x) - 0.5 * threshold), 
                    name='huber_loss')

def epsilon_greedy(action, epsilon, is_action_discrete, action_dim=None):
    if is_action_discrete:
        assert action_dim is not None
        rand_act = tf.random.uniform(
            action.shape, 0, action_dim, dtype=action.dtype)
        action = tf.where(
            tf.random.uniform(action.shape, 0, 1) < epsilon,
            rand_act, action)
    else:
        action = tf.clip_by_value(
            tfd.Normal(action, epsilon).sample(), -1, 1)
    return action

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


def apex_epsilon_greedy(worker_id, envs_per_worker, n_workers, epsilon=.4, alpha=8, sequential=True):
    # the 𝝐-greedy schedule used in Ape-X and Agent57
    assert worker_id is None or worker_id < n_workers, (worker_id, n_workers)
    n_envs = n_workers * envs_per_worker
    env_ids = np.arange(n_envs)
    if worker_id is not None:
        if sequential:
            env_ids = env_ids.reshape((n_workers, envs_per_worker))[worker_id]
        else:
            env_ids = env_ids.reshape((envs_per_worker, n_workers))[:, worker_id]
    env_ids = n_envs - env_ids - 1  # reverse the indices
    return epsilon ** (1 + env_ids / (n_envs - 1) * alpha)


def compute_act_eps(act_eps_type, act_eps, worker_id, n_workers, envs_per_worker):
    assert worker_id is None or worker_id < n_workers, \
        f'worker ID({worker_id}) exceeds range. Valid range: [0, {n_workers})'
    if act_eps_type == 'apex':
        act_eps = apex_epsilon_greedy(
            worker_id, envs_per_worker, n_workers, 
            epsilon=act_eps, 
            sequential=True)
    elif act_eps_type == 'line':
        act_eps = np.linspace(
            0, act_eps, n_workers * envs_per_worker)
        if worker_id is not None:
            act_eps = act_eps.reshape(
                n_workers, envs_per_worker)[worker_id]
    else:
        raise ValueError(f'Unknown type: {act_eps_type}')

    return act_eps


def compute_act_temp(config, model_config, worker_id, n_workers, envs_per_worker):
    if config.get('schedule_act_temp'):
        n_exploit_envs = config.get('n_exploit_envs', 0)
        n_envs = n_workers * envs_per_worker
        n_exploit_envs = config.get('n_exploit_envs')
        if n_exploit_envs:
            act_temps = np.concatenate(
                [np.linspace(config['min_temp'], 1, n_exploit_envs), 
                np.logspace(0, np.log10(config['max_temp']), 
                n_envs - n_exploit_envs+1)[1:]],
                axis=-1)
        else:
            act_temps = np.logspace(
                np.log10(config['min_temp']), np.log10(config['max_temp']), 
                n_workers * envs_per_worker)
        if worker_id is None:
            model_config['actor']['act_temp'] = act_temps
        else:
            act_temps = act_temps.reshape(n_workers, envs_per_worker)
            model_config['actor']['act_temp'] = act_temps[worker_id]
        config['schedule_act_temp'] = False

    return model_config
