import numpy as np


def exp_distance_reward(diff, scale):
  return np.exp(-scale * diff)


