from algo.zero.elements.utils import *

from utility import rl_loss


def compute_joint_stats(
    tape, config, reward, discount, reset, ratio, pi_logprob, 
    value, next_value, gamma, lam, sample_mask
):
    if sample_mask is not None:
        bool_mask = tf.cast(sample_mask, tf.bool)
        ratio = tf.where(bool_mask, ratio, 1.)
        pi_logprob = tf.where(bool_mask, pi_logprob, 0.)
    joint_ratio = tf.math.reduce_prod(ratio, axis=-1)
    joint_pi_logprob = tf.math.reduce_sum(pi_logprob, axis=-1)

    with tape.stop_recording():
        v_target, advantage = rl_loss.compute_target_advantage(
            config=config, 
            reward=tf.reduce_mean(reward, axis=-1), 
            discount=tf.math.reduce_max(discount, axis=-1), 
            reset=tf.gather(reset, 0, axis=-1), 
            value=value, 
            next_value=next_value, 
            ratio=joint_ratio, 
            gamma=gamma, 
            lam=lam, 
            norm_adv=config.get('norm_meta_adv', False)
        )
    
    return joint_ratio, joint_pi_logprob, v_target, advantage
