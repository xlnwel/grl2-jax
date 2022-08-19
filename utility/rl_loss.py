import collections
from typing import Tuple
import tensorflow as tf

from utility import tf_div
from utility.tf_utils import static_scan, reduce_mean, \
    standard_normalization, assert_rank, \
        assert_rank_and_shape_compatibility


DiceCache = collections.namedtuple('DiceCache', 'w deps')
DiceInput = collections.namedtuple('DiceInput', 'logprob lam')

def time_major(*args, axis):
    dims = list(range(args[0].shape.ndims))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        args = [a if a is None else tf.transpose(a, dims) for a in args]
    return dims, args


def to_loss(
    raw_stats, 
    coef, 
    mask=None, 
    n=None, 
):
    is_unit_wise_coef = isinstance(coef, tf.Tensor) and coef.shape != ()
    if is_unit_wise_coef:
        raw_stats = coef * raw_stats
    raw_loss = reduce_mean(raw_stats, mask, n)
    loss = raw_loss if is_unit_wise_coef else coef * raw_loss
    return raw_loss, loss


def magic_box(logprob):
    return tf.exp(logprob - tf.stop_gradient(logprob))


def _dice_impl(
    acc: DiceCache, 
    x: Tuple[tf.Tensor, float], 
):
    v = acc.w * x.lam
    w = v + x.logprob
    deps = magic_box(w) - magic_box(v)
    return DiceCache(w, deps)


def dice(
    logprob, 
    axis: int=None, 
    lam: float=1, 
):
    """ The DiCE operator
    axis: the time dimension. If None, we do not consider the time dimension
    lam: the discount factor to reduce the effect of distant causal dependencies
        so as to trade-off bias and variance.
    exclusive: exclusive cumulative sum
    """
    if axis is not None:
        dims, (logprob, ) = time_major(logprob, axis=axis)
        lam = tf.ones_like(logprob) * lam
        res = static_scan(
            _dice_impl, DiceCache(0, 0), DiceInput(logprob, lam), reverse=False
        )
        if axis != 0:
            deps = tf.transpose(res.deps, dims)
    else:
        deps = magic_box(logprob)
    
    return deps


def huber_loss(x, *, y=None, threshold=1.):
    if y is not None:   # if y is passed, take x-y as error, otherwise, take x as error
        x = x - y
    huber_loss = tf.where(
        tf.abs(x) <= threshold, 
        0.5 * tf.square(x), 
        threshold * (tf.abs(x) - 0.5 * threshold), 
        name='huber_loss'
    )
    return huber_loss


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
    *, 
    reward, 
    value, 
    discount, 
    reset=None, # reset is required for non-episodic environments, where environment resets due to time limits
    gamma, 
    lam, 
    bootstrap=None, 
    axis=0
):
    """
    discount = 1-done. 
    axis specifies the time dimension
    """
    # swap 'axis' with the 0-th dimension
    dims, (reward, value, discount, reset) = \
        time_major(reward, value, discount, reset, axis=axis)
    
    if bootstrap is None:
        bootstrap = tf.zeros_like(value[-1])
    next_value = tf.concat([value[1:], bootstrap[None]], 0)
    
    # 1-step target: r + 𝛾 * v'
    discount = discount * gamma
    inputs = reward + discount * next_value
    if reset is not None:
        discount = (1 - reset) * gamma
    discount = discount * lam

    # lambda function computes lambda return starting from the end
    target = static_scan(
        lambda acc, x: x[0] + x[1] * acc,
        bootstrap, (inputs, discount), reverse=True
    )
    if axis != 0:
        target = tf.transpose(target, dims)

    return target


def retrace(
    *,
    reward, 
    q,  
    next_qs, 
    next_pi, 
    ratio, 
    discount, 
    reset=None, 
    gamma, 
    lam=.95, 
    c_clip=1, 
    axis=0, 
    tbo=False, 
    regularization=None
):
    """
    Params:
        discount = 1-done. 
        axis specifies the time dimension
    """
    assert_rank_and_shape_compatibility(
        [reward, q, next_qs, discount])

    # swap 'axis' with the 0-th dimension
    dims, (next_q, ratio, discount, reset) = \
        time_major(next_q, ratio, discount, reset, axis=axis)

    if tbo:
        next_qs = inverse_h(next_qs)
    next_v = tf.reduce_sum(next_qs * next_pi, axis=-1)
    if regularization is not None:
        next_v -= regularization
    discount = discount * gamma
    delta = reward + discount * next_v - q
    next_c = lam * tf.minimum(ratio[1:], c_clip)
    initial_value = delta[-1]

    assert_rank([delta[:-1], next_c])
    if reset is not None:
        discounted_ratio = (1 - reset[1:]) * gamma * next_c
    else:
        discounted_ratio = discount[1:] * next_c
    diff = static_scan(
        lambda acc, x: x[0] + x[1] * acc,
        initial_value, (delta[:-1], discounted_ratio), reverse=True
    )

    target = diff + q

    if axis != 0:
        target = tf.transpose(target, dims)

    if tbo:
        target = h(target)
        
    return target


def gae(
    *, 
    reward, 
    value, 
    next_value, 
    discount, 
    reset=None, 
    gamma=1, 
    lam=1, 
    norm_adv=False, 
    zero_center=True, 
    epsilon=1e-8, 
    clip=None, 
    mask=None, 
    n=None, 
    axis=0
):
    assert_rank_and_shape_compatibility(
        [reward, value, next_value, discount])
    
    # swap 'axis' with the 0-th dimension
    # to make all tensors time-major
    dims, (reward, value, next_value, discount, reset) = \
        time_major(reward, value, next_value, discount, reset, axis=axis)

    gae_discount = gamma * lam
    delta = reward + discount * gamma * next_value - value
    discount = (discount if reset is None else (1 - reset)) * gae_discount
    initial_value = tf.zeros_like(delta[-1])

    adv = static_scan(
        lambda acc, x: x[0] + x[1] * acc,
        initial_value, (delta, discount), reverse=True
    )
    vs = adv + value

    if norm_adv:
        if mask is not None:
            mask = tf.transpose(mask, dims)
        adv = standard_normalization(
            adv, 
            zero_center=zero_center, 
            mask=mask, 
            n=n, 
            epsilon=epsilon, 
            clip=clip
        )

    if axis != 0:
        vs = tf.transpose(vs, dims)
        adv = tf.transpose(adv, dims)

    return vs, adv

def v_trace(
    *,
    reward, 
    value, 
    next_value, 
    pi, 
    mu, 
    discount, 
    reset=None, 
    gamma=1, 
    lam=1, 
    c_clip=1, 
    rho_clip=1, 
    rho_clip_pg=1, 
    adv_type='vtrace', 
    norm_adv=False, 
    zero_center=True, 
    epsilon=1e-8, 
    clip=None, 
    mask=None, 
    n=None, 
    axis=0
):
    """
    Params:
        discount = 1-done. 
        axis specifies the time dimension
    """
    ratio = pi / mu
    return v_trace_from_ratio(
        reward=reward, 
        value=value, 
        next_value=next_value,
        ratio=ratio, 
        discount=discount, 
        reset=reset, 
        gamma=gamma, 
        lam=lam, 
        c_clip=c_clip, 
        rho_clip=rho_clip, 
        rho_clip_pg=rho_clip_pg, 
        adv_type=adv_type, 
        norm_adv=norm_adv, 
        zero_center=zero_center, 
        epsilon=epsilon, 
        clip=clip, 
        mask=mask, 
        n=n, 
        axis=axis
    )


def v_trace_from_ratio(
    *, 
    reward, 
    value, 
    next_value, 
    ratio, 
    discount, 
    reset=None, 
    gamma=1, 
    lam=1, 
    c_clip=1, 
    rho_clip=1, 
    rho_clip_pg=1, 
    adv_type='vtrace', 
    norm_adv=False, 
    zero_center=True, 
    epsilon=1e-8, 
    clip=None, 
    mask=None, 
    n=None, 
    axis=0
):
    """
    Params:
        discount = 1-done. 
        axis specifies the time dimension
    """
    assert_rank_and_shape_compatibility(
        [reward, value, next_value, ratio, discount])
    
    # swap 'axis' with the 0-th dimension
    # to make all tensors time-major
    dims, (reward, value, next_value, ratio, discount, reset) = \
        time_major(reward, value, next_value, ratio, discount, reset, axis=axis)
    
    clipped_c = ratio if c_clip is None else tf.minimum(ratio, c_clip)
    clipped_rho = ratio if rho_clip is None else tf.minimum(ratio, rho_clip)
    if lam is not None and lam != 1:
        clipped_c = lam * clipped_c

    discount = discount * gamma
    delta = clipped_rho * (reward + discount * next_value - value)
    if reset is not None:
        discounted_ratio = (1 - reset) * gamma * clipped_c
    else:
        discounted_ratio = discount * clipped_c
    initial_value = tf.zeros_like(delta[-1])

    v_minus_V = static_scan(
        lambda acc, x: x[0] + x[1] * acc,
        initial_value, (delta, discounted_ratio), reverse=True
    )
    
    vs = v_minus_V + value

    next_vs = tf.concat([vs[1:], next_value[-1:]], axis=0)
    clipped_rho_pg = ratio if rho_clip_pg is None else tf.minimum(ratio, rho_clip_pg)
    if adv_type == 'vtrace':
        adv = clipped_rho_pg * (reward + discount * next_vs - value)
    elif adv_type == 'gae':
        adv = v_minus_V

    if norm_adv:
        if mask is not None:
            mask = tf.transpose(mask, dims)
        adv = standard_normalization(
            adv, 
            zero_center=zero_center, 
            mask=mask, 
            n=n, 
            epsilon=epsilon, 
            clip=clip
        )

    if axis != 0:
        vs = tf.transpose(vs, dims)
        adv = tf.transpose(adv, dims)
    
    return vs, adv


def pg_loss(
    *, 
    pg_coef, 
    advantage, 
    logprob, 
    ratio=1., 
    mask=None, 
    n=None
):
    raw_pg = advantage * ratio * logprob
    raw_pg_loss, pg_loss = to_loss(
        -raw_pg, 
        pg_coef, 
        mask=mask, 
        n=n
    )

    return raw_pg_loss, pg_loss


def entropy_loss(
    *, 
    entropy_coef, 
    entropy, 
    mask=None, 
    n=None
):
    raw_entropy_loss, entropy_loss = to_loss(
        -entropy, 
        entropy_coef, 
        mask=mask, 
        n=n
    )

    return raw_entropy_loss, entropy_loss


def ppo_loss(
    *, 
    pg_coef, 
    advantage, 
    ratio, 
    clip_range, 
    mask=None, 
    n=None,
):
    if mask is not None and n is None:
        n = tf.reduce_sum(mask)
        assert_rank_and_shape_compatibility([advantage, mask])
    assert_rank_and_shape_compatibility([ratio, advantage])
    pg_loss, clipped_loss = _compute_ppo_policy_losses(
        advantage, ratio, clip_range)

    raw_ppo = tf.maximum(pg_loss, clipped_loss)
    raw_ppo_loss, ppo_loss = to_loss(
        raw_ppo, 
        pg_coef, 
        mask=mask, 
        n=n
    )

    # We still count how much will be clipped by range .2 when clipping is off
    if clip_range is None:
        clip_range = .2
    clip_frac = reduce_mean(tf.cast(tf.greater(
        tf.abs(ratio - 1.), clip_range), ratio.dtype), mask, n)

    return pg_loss, clipped_loss, raw_ppo_loss, ppo_loss, clip_frac


def high_order_ppo_loss(
    *, 
    pg_coef, 
    advantage, 
    ratio, 
    dice_op, 
    clip_range, 
    mask=None, 
    n=None,
):
    ratio = tf.stop_gradient(ratio)
    tf.debugging.assert_near(dice_op, 1.)
    if mask is not None and n is None:
        n = tf.reduce_sum(mask)
        assert_rank_and_shape_compatibility([advantage, mask])
    assert_rank_and_shape_compatibility([ratio, dice_op, advantage])
    if clip_range is None:
        clip_mask = tf.zeros_like(ratio, dtype=tf.bool)
    else:
        clip_mask = tf.greater(tf.abs(ratio - 1.), clip_range)
    neg_adv = -advantage
    pg_loss = neg_adv * ratio * dice_op
    clipped_loss = neg_adv * tf.where(clip_mask, 1., dice_op) \
        * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)

    raw_ppo = tf.maximum(pg_loss, clipped_loss)
    raw_ppo_loss, ppo_loss = to_loss(
        raw_ppo, 
        pg_coef, 
        mask=mask, 
        n=n
    )

    clip_frac = reduce_mean(tf.cast(clip_mask, ratio.dtype), mask, n)

    return pg_loss, clipped_loss, raw_ppo_loss, ppo_loss, clip_frac


def joint_ppo_loss(
    *, 
    pg_coef, 
    advantage, 
    ratio, 
    clip_range, 
    mask=None, 
    n=None,
):
    if mask is not None and n is None:
        mask = tf.math.reduce_prod(mask, axis=-1)
        n = tf.reduce_sum(mask)
        assert_rank_and_shape_compatibility([advantage, mask])
    joint_ratio = tf.math.reduce_prod(ratio, axis=-1)
    clipped_ratio = tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
    joint_clipped_ratio = tf.math.reduce_prod(clipped_ratio, axis=-1)
    assert_rank_and_shape_compatibility([joint_ratio, advantage])
    neg_adv = -advantage
    pg_loss = neg_adv * joint_ratio
    clipped_loss = neg_adv * joint_clipped_ratio

    raw_ppo = tf.maximum(pg_loss, clipped_loss)
    raw_ppo_loss, ppo_loss = to_loss(
        raw_ppo, 
        pg_coef, 
        mask=mask, 
        n=n
    )

    # We still count how much will be clipped by range .2 when clipping is off
    if clip_range is None:
        clip_range = .2
    clip_frac = reduce_mean(tf.cast(tf.greater(
        tf.abs(ratio - 1.), clip_range), ratio.dtype), mask, n)

    return pg_loss, clipped_loss, raw_ppo_loss, ppo_loss, clip_frac


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


def clipped_value_loss(
    value, 
    traj_ret, 
    old_value, 
    clip_range, 
    mask=None, 
    n=None, 
    huber_threshold=None,
):
    assert_rank_and_shape_compatibility([value, traj_ret, old_value])
    value_diff, loss1, loss2 = _compute_ppo_value_losses(
        value, traj_ret, old_value, clip_range, huber_threshold)
    
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


def _compute_ppo_policy_losses(advantages, ratio, clip_range):
    neg_adv = -advantages
    loss1 = neg_adv * ratio
    if clip_range is None:
        loss2 = loss1
    else:
        loss2 = neg_adv * tf.clip_by_value(ratio, 1. - clip_range, 1. + clip_range)
    return loss1, loss2


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
        raw_kl_loss, kl_loss = to_loss(
            kl, 
            kl_coef, 
            mask=sample_mask, 
            n=n
        )
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
        raw_js_loss, js_loss = to_loss(
            js, 
            js_coef, 
            mask=sample_mask, 
            n=n
        )
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
        raw_tsallis_loss, tsallis_loss = to_loss(
            tsallis, 
            tsallis_coef, 
            mask=sample_mask, 
            n=n
        )
        tf.debugging.assert_all_finite(raw_tsallis_loss, 'Bad raw_tsallis_loss')
    else:
        tsallis = 0.
        raw_tsallis_loss = 0.
        tsallis_loss = 0.

    return tsallis, raw_tsallis_loss, tsallis_loss


def dr_loss(
    reward, 
    value, 
    next_value, 
    ratio, 
    discount, 
    reset=None, 
    gamma=1, 
    lam=1, 
    c_clip=1, 
    rho_clip=1, 
    rho_clip_pg=1, 
    axis=0
):
    pass