import tensorflow as tf
from tensorflow_probability import distributions as tfd


def kl_from_distributions(
    *,
    pi1=None, 
    pi2=None,
    pi1_mean=None,  
    pi1_std=None,  
    pi2_mean=None,  
    pi2_std=None,  
    pi_mask=None,
):
    if pi1 is None:
        d1 = tfd.MultivariateNormalDiag(pi1_mean, pi1_std)
        d2 = tfd.MultivariateNormalDiag(pi2_mean, pi2_std)
        kl = d1.kl_divergence(d2)
    else:
        log_pi1 = tf.math.log(tf.clip_by_value(pi1, 1e-10, 1))
        log_pi2 = tf.math.log(tf.clip_by_value(pi2, 1e-10, 1))
        log_ratio = log_pi1 - log_pi2
        if pi_mask is not None:
            log_ratio = tf.where(pi_mask, log_ratio, 0)
        tf.debugging.assert_all_finite(log_ratio, 'Bad log_ratio')
        kl = tf.reduce_sum(pi1 * log_ratio, axis=-1)

    return kl

def kl_from_samples(
    *,
    logp,
    logq, 
    sample_prob, 
):
    log_ratio = logp - logq
    approx_kl = -tf.exp(logq) * tf.stop_gradient(
        tf.sign(log_ratio) * tf.exp(log_ratio) / sample_prob)
    return approx_kl

def reverse_kl_from_samples(
    *,
    logp, 
    logq, 
    sample_prob
):
    log_ratio = logp - logq
    approx_kl = tf.exp(logp) \
        * tf.stop_gradient(log_ratio / sample_prob)
    return approx_kl

def js_from_samples(
    *,
    p,
    q, 
    sample_prob
):
    q = tf.clip_by_value(q, 1e-10, 1)
    p_plus_q = tf.clip_by_value(p+q, 1e-10, 1)
    approx_js = .5 * q * tf.stop_gradient(
        (tf.math.log(2.) + tf.math.log(q) -  tf.math.log(p_plus_q)) / sample_prob)
    return approx_js

def js_from_distributions(
    *,
    pi1,
    pi2, 
    pi_mask=None
):
    avg = (pi1 + pi2) / 2
    approx_js = .5 * (
        kl_from_distributions(pi1=pi1, pi2=avg, pi_mask=pi_mask)
        + kl_from_distributions(pi1=pi2, pi2=avg, pi_mask=pi_mask)
    )
    return approx_js

def tv_from_samples(
    *, 
    p,
    q, 
    sample_prob
):
    approx_tv = .5 * tf.abs(p-q) / sample_prob
    return approx_tv

def tsallis_log(p, tsallis_q):
    p = tf.clip_by_value(p, 1e-10, 1)
    if tsallis_q == 1:
        return tf.math.log(p)
    else:
        return (p**(1-tsallis_q) - 1) / (1 - tsallis_q)

def tsallis_exp(p, tsallis_q):
    if tsallis_q == 1:
        return tf.math.exp(p)
    else:
        return tf.maximum(
            0, 1 + (1-tsallis_q) * p)**(1 / (1-tsallis_q))

def tsallis_from_samples(
    *, 
    p, 
    q, 
    sample_prob, 
    tsallis_q, 
):
    approx_tsallis = q * tf.stop_gradient(
        tf.sign(q-p) * p * q ** (-tsallis_q) / sample_prob)

    return approx_tsallis

def reverse_tsallis_from_samples(
    *, 
    p, 
    q, 
    sample_prob, 
    tsallis_q,
):
    logp = tsallis_log(p, tsallis_q)
    logq = tsallis_log(q, tsallis_q)
    approx_tsallis = p * tf.stop_gradient(
        -tsallis_q * p ** (tsallis_q - 1) * (logq - logp) / sample_prob)
    return approx_tsallis

def tsallis_from_distributions(
    *,
    pi1=None, 
    pi2=None,
    pi1_mean=None,  
    pi1_std=None,  
    pi2_mean=None,  
    pi2_std=None,  
    pi_mask=None,
    tsallis_q, 
):
    if pi1 is None:
        raise NotImplementedError('Tsallis divergence only support discrete probability distributions')
    else:
        log_pi1 = tsallis_log(tf.clip_by_value(pi1, 1e-10, 1), tsallis_q)
        log_pi2 = tsallis_log(tf.clip_by_value(pi2, 1e-10, 1), tsallis_q)
        log_ratio = log_pi1 - log_pi2
        if pi_mask is not None:
            log_ratio = tf.where(pi_mask, log_ratio, 0)
        tf.debugging.assert_all_finite(log_ratio, 'Bad log_ratio')
        tsallis = tf.reduce_sum(pi1**tsallis_q * log_ratio, axis=-1)

    return tsallis
