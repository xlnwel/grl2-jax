import collections


ModelWeights = collections.namedtuple('model_weights', 'model weights')
ModelStats = collections.namedtuple('model_stats', 'model stats')

class Status:
  TRAINING = 'training'
  SCORE_MET = 'score_met'
  TIMEOUT = 'timeout'
