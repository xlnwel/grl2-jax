from tools.utils import except_axis
from algo.ma_common.elements.utils import *


def norm_adv(
  config, 
  raw_adv, 
  sample_mask=None, 
  n=None, 
  epsilon=1e-5
):
  if config.norm_adv:
    advantage = jax_math.standard_normalization(
      raw_adv, 
      zero_center=config.get('zero_center', True), 
      mask=sample_mask, 
      n=n, 
      axis=except_axis(raw_adv, UNIT_DIM), 
      epsilon=epsilon, 
    )
  else:
    advantage = raw_adv
  advantage = lax.stop_gradient(advantage)
  return advantage
