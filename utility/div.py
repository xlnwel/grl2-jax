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
    approx_js = .5 * q * tf.stop_gradient(
        (tf.math.log(2.) + tf.math.log(q) -  tf.math.log(p+q)) / sample_prob)
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

def tsallis_from_samples(
    *, 
    p, 
    q, 
    sample_prob, 
    tsallis_q,
):
    if tsallis_q == 1:
        logp = tf.math.log(p)
        logq = tf.math.log(q)
        return reverse_kl_from_samples(
            logp=logp, logq=logq, sample_prob=sample_prob)
    else:
        pass