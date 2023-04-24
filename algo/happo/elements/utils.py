from tools.utils import except_axis
from algo.ppo.elements.utils import *


def norm_adv(
    config, 
    raw_adv, 
    teammate_log_ratio, 
    teammate_ratio_clip=None, 
    sample_mask=None, 
    n=None, 
    epsilon=1e-5
):
    if config.norm_adv:
        norm_adv = jax_math.standard_normalization(
            raw_adv, 
            zero_center=config.get('zero_center', True), 
            mask=sample_mask, 
            n=n, 
            axis=except_axis(raw_adv, UNIT_DIM), 
            epsilon=epsilon, 
        )
    else:
        norm_adv = raw_adv
    if norm_adv.ndim < teammate_log_ratio.ndim:
        norm_adv = jnp.expand_dims(norm_adv, -1)
    jax_assert.assert_rank_compatibility(
        [norm_adv, teammate_log_ratio])
    tm_ratio = lax.exp(teammate_log_ratio)
    if teammate_ratio_clip is not None:
        tm_ratio = jnp.clip(tm_ratio, 1-teammate_ratio_clip, 1+teammate_ratio_clip)
    advantage = norm_adv * tm_ratio
    advantage = lax.stop_gradient(advantage)
    return norm_adv, advantage
