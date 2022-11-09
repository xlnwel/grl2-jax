from jax import lax
import jax.numpy as jnp
import rlax
import chex


""" TODO: applying mask to logits instead of probs """
def kl_from_distributions(
    *, 
    p_logits=None, 
    q_logits=None, 
    p_mean=None, 
    q_mean=None, 
    p_std=None,  
    q_std=None,  
):
    if p_logits is None:
        kl = rlax.multivariate_normal_kl_divergence(
            p_mean, p_std, q_mean, q_std)
    else:
        kl = rlax.categorical_kl_divergence(p_logits, q_logits)

    return kl

def kl_from_samples(
    *,
    logp,
    logq, 
    sample_prob, 
):
    if sample_prob is None:
        sample_prob = 1.
    log_ratio = logp - logq
    p = lax.exp(logp)
    approx_kl = p * lax.stop_gradient((log_ratio + 1) / sample_prob)
    return approx_kl

def reverse_kl_from_samples(
    *,
    logp, 
    logq, 
    sample_prob
):
    if sample_prob is None:
        sample_prob = 1.
    p = lax.exp(logp)
    q = lax.exp(logq)
    approx_kl = q * lax.stop_gradient(p / q / sample_prob)
    return approx_kl

def js_from_samples(
    *,
    p,
    q, 
    sample_prob
):
    q = jnp.clip(q, 1e-10, 1)
    p_plus_q = jnp.clip(p+q, 1e-10, 1)
    approx_js = .5 * q * lax.stop_gradient(
        (lax.log(2.) + lax.log(q) -  lax.log(p_plus_q)) / sample_prob)
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
    approx_tv = .5 * lax.abs(p-q) / sample_prob
    return approx_tv

def tsallis_log(p, tsallis_q):
    p = jnp.clip(p, 1e-10, 1)
    if tsallis_q == 1:
        return lax.log(p)
    else:
        return (p**(1-tsallis_q) - 1) / (1 - tsallis_q)

def tsallis_exp(p, tsallis_q):
    if tsallis_q == 1:
        return lax.exp(p)
    else:
        return jnp.maximum(
            0, 1 + (1-tsallis_q) * p)**(1 / (1-tsallis_q))

def tsallis_from_samples(
    *, 
    p, 
    q, 
    sample_prob, 
    tsallis_q, 
):
    approx_tsallis = q * lax.stop_gradient(
        lax.sign(q-p) * p * q ** (-tsallis_q) / sample_prob)

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
    approx_tsallis = p * lax.stop_gradient(
        -tsallis_q * p ** (tsallis_q - 1) * (logq - logp) / sample_prob)
    return approx_tsallis

def tsallis_from_distributions(
    *,
    pi1=None, 
    pi2=None,
    p_mean=None,  
    p_std=None,  
    q_mean=None,  
    q_std=None,  
    pi_mask=None,
    tsallis_q, 
):
    if pi1 is None:
        raise NotImplementedError('Tsallis divergence only support discrete probability distributions')
    else:
        log_pi1 = tsallis_log(jnp.clip(pi1, 1e-10, 1), tsallis_q)
        log_pi2 = tsallis_log(jnp.clip(pi2, 1e-10, 1), tsallis_q)
        log_ratio = log_pi1 - log_pi2
        if pi_mask is not None:
            log_ratio = jnp.where(pi_mask, log_ratio, 0)
        chex.assert_tree_all_finite(log_ratio)
        tsallis = jnp.sum(pi1**tsallis_q * log_ratio, axis=-1)

    return tsallis
