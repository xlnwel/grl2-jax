import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility import tf_div
from utility.tf_utils import static_scan, reduce_mean, \
    assert_rank, assert_rank_and_shape_compatibility


def huber_loss(x, *, y=None, threshold=1.):
    if y is not None:   # if y is passed, take x-y as error, otherwise, take x as error
        x = x - y
    return tf.where(tf.abs(x) <= threshold, 
                    0.5 * tf.square(x), 
                    threshold * (tf.abs(x) - 0.5 * threshold), 
                    name='huber_loss')


def quantile_regression_loss(
    qtv, 
    target, 
    tau_hat, 
    kappa=1., 
    return_error=False
):
    assert qtv.shape[-1] == 1, qtv.shape
    assert target.shape[-2] == 1, target.shape
    assert tau_hat.shape[-1] == 1, tau_hat.shape
    assert_rank([qtv, target, tau_hat])
    error = target - qtv           # [B, N, N']
    weight = tf.abs(tau_hat - tf.cast(error < 0, tf.float32))   # [B, N, N']
    huber = huber_loss(error, threshold=kappa)                  # [B, N, N']
    qr_loss = tf.reduce_sum(tf.reduce_mean(weight * huber, axis=-1), axis=-2) # [B]

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


def n_step_target(
    reward, 
    nth_value, 
    discount=1., 
    gamma=.99, 
    steps=1., 
    tbo=False
):
    """
    discount is only the done signal
    """
    if tbo:
        return h(reward + discount * gamma**steps * inverse_h(nth_value))
    else:
        return reward + discount * gamma**steps * nth_value


def lambda_return(
    reward, 
    value, 
    discount, 
    lambda_, 
    bootstrap=None, 
    axis=0
):
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


def retrace(
    reward, 
    next_qs, 
    next_action, 
    next_pi, 
    next_mu_a, 
    discount, 
    lambda_=.95, 
    ratio_clip=1, 
    axis=0, 
    tbo=False, 
    regularization=None
):
    """
    Params:
        discount = gamma * (1-done). 
        axis specifies the time dimension
    """
    if isinstance(discount, (int, float)):
        discount = discount * tf.ones_like(reward)
    if next_action.dtype.is_integer:
        next_action = tf.one_hot(next_action, next_pi.shape[-1], dtype=next_pi.dtype)
    assert_rank_and_shape_compatibility([next_action, next_pi], reward.shape.ndims + 1)
    next_pi_a = tf.reduce_sum(next_pi * next_action, axis=-1)
    next_ratio = next_pi_a / next_mu_a
    if ratio_clip is not None:
        next_ratio = tf.minimum(next_ratio, ratio_clip)
    next_c = next_ratio * lambda_

    if tbo:
        next_qs = inverse_h(next_qs)
    next_v = tf.reduce_sum(next_qs * next_pi, axis=-1)
    if regularization is not None:
        next_v -= regularization
    next_q = tf.reduce_sum(next_qs * next_action, axis=-1)
    current = reward + discount * (next_v - next_c * next_q)

    # swap 'axis' with the 0-th dimension
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        next_q = tf.transpose(next_q, dims)
        current = tf.transpose(current, dims)
        discount = tf.transpose(discount, dims)
        next_c = tf.transpose(next_c, dims)

    assert_rank([current, discount, next_c])
    target = static_scan(
        lambda acc, x: x[0] + x[1] * x[2] * acc,
        next_q[-1], (current, discount, next_c), 
        reverse=True)

    if axis != 0:
        target = tf.transpose(target, dims)

    if tbo:
        target = h(target)
        
    return target


def v_trace(
    reward, 
    value, 
    next_value, 
    pi1, 
    mu, 
    discount, 
    lambda_=1, 
    c_clip=1, 
    rho_clip=1, 
    rho_clip_pg=1, 
    axis=0
):
    """
    Params:
        discount = gamma * (1-done). 
        axis specifies the time dimension
    """
    ratio = pi1 / mu
    return v_trace_from_ratio(reward, value, next_value, 
        ratio, discount, lambda_, c_clip, 
        rho_clip, rho_clip_pg, axis)


def v_trace_from_ratio(
    reward, 
    value, 
    next_value, 
    ratio, 
    discount, 
    lambda_=1, 
    c_clip=1, 
    rho_clip=1, 
    rho_clip_pg=1, 
    axis=0
):
    """
    Params:
        discount = gamma * (1-done). 
        axis specifies the time dimension
    """
    assert_rank_and_shape_compatibility(
        [reward, value, next_value, ratio, discount])
    
    # swap 'axis' with the 0-th dimension
    # to make all tensors time-major
    dims = list(range(reward.shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = tf.transpose(reward, dims)
        value = tf.transpose(value, dims)
        next_value = tf.transpose(next_value, dims)
        ratio = tf.transpose(ratio, dims)
        discount = tf.transpose(discount, dims)
    
    clipped_c = ratio if c_clip is None else tf.minimum(ratio, c_clip)
    clipped_rho = ratio if rho_clip is None else tf.minimum(ratio, rho_clip)
    if lambda_ is not None and lambda_ != 1:
        clipped_rho = lambda_ * clipped_rho

    delta = clipped_rho * (reward + discount * next_value - value)
    
    initial_value = tf.zeros_like(delta[-1])

    v_minus_V = static_scan(
        lambda acc, x: x[0] + x[1] * x[2] * acc,
        initial_value, (delta, discount, clipped_c),
        reverse=True)
    
    vs = v_minus_V + value

    next_vs = tf.concat([vs[1:], next_value[-1:]], axis=0)
    clipped_rho_pg = ratio if rho_clip_pg is None else tf.minimum(ratio, rho_clip_pg)
    adv = clipped_rho_pg * (reward + discount * next_vs - value)

    if axis != 0:
        vs = tf.transpose(vs, dims)
        adv = tf.transpose(adv, dims)
    
    return vs, adv


def ppo_loss(
    *, 
    pg_coef, 
    entropy_coef, 
    log_ratio, 
    advantage, 
    clip_range, 
    entropy, 
    mask=None, 
    n=None,
):
    is_unit_wise_coef = isinstance(pg_coef, tf.Tensor) and pg_coef.shape != ()
    if mask is not None and n is None:
        n = tf.reduce_sum(mask)
        assert_rank_and_shape_compatibility([advantage, mask])
    assert_rank_and_shape_compatibility([log_ratio, advantage])
    ratio, pg_loss, clipped_loss = _compute_ppo_policy_losses(
        log_ratio, advantage, clip_range)

    raw_ppo = tf.maximum(pg_loss, clipped_loss)
    if is_unit_wise_coef:
        raw_ppo = pg_coef * raw_ppo
    tf.debugging.assert_all_finite(raw_ppo, 'Bad raw_ppo')
    raw_ppo_loss = reduce_mean(raw_ppo, mask=mask, n=n)
    ppo_loss = raw_ppo_loss if is_unit_wise_coef else pg_coef * raw_ppo_loss
    is_unit_wise_coef = isinstance(entropy_coef, tf.Tensor) and entropy_coef.shape != ()
    if is_unit_wise_coef:
        entropy = entropy_coef * entropy
    raw_entropy_loss = - reduce_mean(entropy, mask, n)
    entropy_loss = raw_entropy_loss if is_unit_wise_coef else entropy_coef * raw_entropy_loss

    # debug stats: KL between old and current policy and fraction of data being clipped
    approx_kl = .5 * reduce_mean((-log_ratio)**2, mask, n)
    # We still count how much will be clipped by range .2 when clipping is off
    if clip_range is None:
        clip_range = .2
    clip_frac = reduce_mean(tf.cast(tf.greater(
        tf.abs(ratio - 1.), clip_range), ratio.dtype), mask, n)

    return ratio, pg_loss, clipped_loss, raw_ppo_loss, ppo_loss, \
        raw_entropy_loss, entropy_loss, approx_kl, clip_frac


def clipped_value_loss(
    value, 
    traj_ret, 
    old_value, 
    clip_range, 
    mask=None, 
    n=None, 
    huber_threshold=None,
    reduce=True,
):
    if mask is not None and n is None:
        n = tf.reduce_sum(mask)
        assert_rank_and_shape_compatibility([value, mask])
    assert_rank_and_shape_compatibility([value, traj_ret, old_value])
    value_diff, loss1, loss2 = _compute_ppo_value_losses(
        value, traj_ret, old_value, clip_range, huber_threshold)
    
    if reduce:
        value_loss = reduce_mean(tf.maximum(loss1, loss2), mask, n)
    else:
        value_loss = tf.maximum(loss1, loss2)
    clip_frac = reduce_mean(
        tf.cast(tf.greater(tf.abs(value_diff), clip_range), value.dtype), mask, n)

    return value_loss, clip_frac


def tppo_loss(
    log_ratio, 
    kl, 
    advantages, 
    kl_weight, 
    clip_range, 
    entropy
):
    ratio = tf.exp(log_ratio)
    condition = tf.math.logical_and(
        kl >= clip_range, ratio * advantages > advantages)
    objective = tf.where(
        condition,
        ratio *  advantages - kl_weight * kl,
        ratio * advantages
    )

    policy_loss = -tf.reduce_mean(objective)
    clip_frac = tf.reduce_mean(tf.cast(condition, tf.float32))
    entropy = tf.reduce_mean(entropy)

    return policy_loss, entropy, clip_frac


def _compute_ppo_policy_losses(log_ratio, advantages, clip_range):
    ratio = tf.exp(log_ratio)
    neg_adv = -advantages
    loss1 = neg_adv * ratio
    if clip_range is None:
        loss2 = loss1
    else:
        loss2 = neg_adv * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
    return ratio, loss1, loss2


def _compute_ppo_value_losses(
    value, 
    traj_ret, 
    old_value, 
    clip_range, 
    huber_threshold=None
):
    value_diff = value - old_value
    value_clipped = old_value + tf.clip_by_value(value_diff, -clip_range, clip_range)
    if huber_threshold is None:
        loss1 = .5 * tf.square(value - traj_ret)
        loss2 = .5 * tf.square(value_clipped - traj_ret)
    else:
        loss1 = huber_loss(value, y=traj_ret, threshold=huber_threshold)
        loss2 = huber_loss(value_clipped, y=traj_ret, threshold=huber_threshold)

    return value_diff, loss1, loss2


def compute_kl(
    *, 
    kl_type,
    kl_coef=None, 
    logp=None,
    logq=None, 
    sample_prob=None, 
    pi1=None,
    pi2=None,
    pi1_mean=None,
    pi2_mean=None,
    pi1_std=None,
    pi2_std=None,
    pi_mask=None, 
    sample_mask=None,
    n=None, 
):
    is_unit_wise_coef = isinstance(kl_coef, tf.Tensor) and kl_coef.shape != ()
    if kl_coef is not None:
        if kl_type == 'forward_approx':
            kl = tf_div.kl_from_samples(
                logp=logp, 
                logq=logq,
                sample_prob=sample_prob, 
            )
        elif kl_type == 'reverse_approx':
            kl = tf_div.reverse_kl_from_samples(
                logp=logq, 
                logq=logp,
                sample_prob=sample_prob, 
            )
        elif kl_type == 'forward':
            kl = tf_div.kl_from_distributions(
                pi1=pi1, 
                pi2=pi2, 
                pi1_mean=pi1_mean, 
                pi1_std=pi1_std, 
                pi2_mean=pi2_mean, 
                pi2_std=pi2_std, 
                pi_mask=pi_mask
            )
        elif kl_type == 'reverse':
            kl = tf_div.kl_from_distributions(
                pi1=pi2, 
                pi2=pi1, 
                pi1_mean=pi2_mean, 
                pi1_std=pi2_std, 
                pi2_mean=pi1_mean,
                pi2_std=pi1_std, 
                pi_mask=pi_mask
            )
        else:
            raise NotImplementedError(f'Unknown kl {kl_type}')
        if is_unit_wise_coef:
            kl = kl_coef * kl
        raw_kl_loss = reduce_mean(kl, mask=sample_mask, n=n)
        tf.debugging.assert_all_finite(raw_kl_loss, 'Bad raw_kl_loss')
        kl_loss = raw_kl_loss if is_unit_wise_coef else kl_coef * raw_kl_loss
    else:
        kl = 0.
        raw_kl_loss = 0.
        kl_loss = 0.

    return kl, raw_kl_loss, kl_loss


def compute_js(
    *, 
    js_type, 
    js_coef, 
    p=None, 
    q=None,
    sample_prob=None, 
    pi1=None,
    pi2=None,
    pi_mask=None, 
    sample_mask=None, 
    n=None
):
    is_unit_wise_coef = isinstance(js_coef, tf.Tensor) and js_coef.shape != ()
    if js_coef is not None:
        if js_type == 'approx':
            js = tf_div.js_from_samples(
                p=p, 
                q=q, 
                sample_prob=sample_prob, 
            )
        elif js_type == 'exact':
            js = tf_div.js_from_distributions(
                pi1=pi1, pi2=pi2, pi_mask=pi_mask
            )
        else:
            raise NotImplementedError(f'Unknown JS type {js_type}')
        if is_unit_wise_coef:
            js = js_coef * js
        raw_js_loss = reduce_mean(js, mask=sample_mask, n=n)
        tf.debugging.assert_all_finite(raw_js_loss, 'Bad raw_js_loss')
        js_loss = raw_js_loss if is_unit_wise_coef else js_coef * raw_js_loss
    else:
        js = 0,
        raw_js_loss = 0.
        js_loss = 0.
    
    return js, raw_js_loss, js_loss


def compute_tsallis(
    *, 
    tsallis_type,
    tsallis_coef, 
    tsallis_q, 
    p=None,
    q=None, 
    sample_prob=None, 
    pi1=None,
    pi2=None,
    pi1_mean=None,
    pi2_mean=None,
    pi1_std=None,
    pi2_std=None,
    pi_mask=None, 
    sample_mask=None,
    n=None, 
):
    is_unit_wise_coef = isinstance(tsallis_coef, tf.Tensor) and tsallis_coef.shape != ()
    if tsallis_coef is not None:
        if tsallis_type == 'forward_approx':
            tsallis = tf_div.tsallis_from_samples(
                p=p, 
                q=q,
                sample_prob=sample_prob, 
                tsallis_q=tsallis_q, 
            )
        elif tsallis_type == 'reverse_approx':
            tsallis = tf_div.reverse_tsallis_from_samples(
                p=q, 
                q=p,
                sample_prob=sample_prob, 
                tsallis_q=tsallis_q, 
            )
        elif tsallis_type == 'forward':
            tsallis = tf_div.tsallis_from_distributions(
                pi1=pi1, 
                pi2=pi2, 
                pi1_mean=pi1_mean, 
                pi1_std=pi1_std, 
                pi2_mean=pi2_mean, 
                pi2_std=pi2_std, 
                pi_mask=pi_mask, 
                tsallis_q=tsallis_q, 
            )
        elif tsallis_type == 'reverse':
            tsallis = tf_div.tsallis_from_distributions(
                pi1=pi2, 
                pi2=pi1, 
                pi1_mean=pi2_mean, 
                pi1_std=pi2_std, 
                pi2_mean=pi1_mean,
                pi2_std=pi1_std, 
                pi_mask=pi_mask, 
                tsallis_q=tsallis_q
            )
        else:
            raise NotImplementedError(f'Unknown Tsallis {tsallis_type}')
        if is_unit_wise_coef:
            tsallis = tsallis_coef * tsallis
        raw_tsallis_loss = reduce_mean(tsallis, mask=sample_mask, n=n)
        tf.debugging.assert_all_finite(raw_tsallis_loss, 'Bad raw_tsallis_loss')
        tsallis_loss = raw_tsallis_loss if is_unit_wise_coef else tsallis_coef * raw_tsallis_loss
    else:
        tsallis = 0.
        raw_tsallis_loss = 0.
        tsallis_loss = 0.

    return tsallis, raw_tsallis_loss, tsallis_loss
