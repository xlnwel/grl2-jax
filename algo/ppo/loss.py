import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy


def _reduce_mean(x, n):
    return tf.reduce_mean(x) if n is None else tf.reduce_sum(x) / n

def compute_ppo_loss(log_ratio, advantages, clip_range, entropy):
    with tf.name_scope('ppo_loss'):
        ratio = tf.exp(log_ratio)
        loss1 = -advantages * ratio
        loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
        
        ppo_loss = tf.reduce_mean(tf.maximum(loss1, loss2))
        entropy = tf.reduce_mean(entropy)
        # debug stats: KL between old and current policy and fraction of data being clipped
        approx_kl = .5 * tf.reduce_mean(tf.square(-log_ratio))
        p_clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), tf.float32))
    return ppo_loss, entropy, approx_kl, p_clip_frac

def compute_value_loss(value, traj_ret, old_value, clip_range):
    with tf.name_scope('value_loss'):
        value_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
        loss1 = tf.square(value - traj_ret)
        loss2 = tf.square(value_clipped - traj_ret)
        
        value_loss = .5 * tf.reduce_mean(tf.maximum(loss1, loss2))
        v_clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(value-old_value), clip_range), tf.float32))

    return value_loss, v_clip_frac

def compute_ppo_loss_with_mask(log_ratio, advantages, clip_range, entropy, mask=None, n=None):
    assert (mask is None) == (n is None), \
        f'Both/Neither mask and/nor n should be None, but get \nmask:{mask}\nn:{n}'
    dtype = global_policy().compute_dtype
    
    m = 1. if mask is None else mask
    with tf.name_scope('ppo_loss'):
        ratio = tf.exp(log_ratio)
        loss1 = -advantages * ratio
        loss2 = -advantages * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
        
        ppo_loss = _reduce_mean(tf.maximum(loss1, loss2) * m, n)
        entropy = _reduce_mean(entropy * m, n)
        # debug stats: KL between old and current policy and fraction of data being clipped
        approx_kl = .5 * _reduce_mean((-log_ratio)**2 * m, n)
        p_clip_frac = _reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.), clip_range), dtype) * m, n)
    return ppo_loss, entropy, approx_kl, p_clip_frac

def compute_value_loss_with_mask(value, traj_ret, old_value, clip_range, mask=None, n=None):
    assert (mask is None) == (n is None), \
        f'Both/Neither mask and/nor n should be None, but get \nmask:{mask}\nn:{n}'
    dtype = global_policy().compute_dtype

    m = 1. if mask is None else mask
    with tf.name_scope('value_loss'):
        value_clipped = old_value + tf.clip_by_value(value - old_value, -clip_range, clip_range)
        loss1 = (value - traj_ret)**2
        loss2 = (value_clipped - traj_ret)**2
        
        value_loss = .5 * _reduce_mean(tf.maximum(loss1, loss2) * m, n)
        v_clip_frac = _reduce_mean(
            tf.cast(tf.greater(tf.abs(value-old_value), clip_range), dtype) * m, n)

    return value_loss, v_clip_frac
