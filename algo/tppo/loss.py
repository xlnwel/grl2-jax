from algo.ppo.loss import *


def compute_tppo_loss(log_ratio, kl, advantages, kl_weight, clip_range, entropy):
    with tf.name_scope('tppo_loss'):
        ratio = tf.exp(log_ratio)
        condition = tf.math.logical_and(kl > clip_range, ratio * advantages > advantages)
        objective = tf.where(
            condition,
            ratio *  advantages - kl_weight * kl,
            ratio * advantages
        )

        tppo_loss = -tf.reduce_mean(objective)
        clip_frac = tf.reduce_mean(tf.cast(condition, tf.float32))
        entropy = tf.reduce_mean(entropy)

    return tppo_loss, entropy, clip_frac
