import collections


ModelWeights = collections.namedtuple('model_weights', 'model weights')
ModelStats = collections.namedtuple('model_stats', 'model stats')
LoggedStats = collections.namedtuple('logged_stats', 'path type stats')


class LoggedType:
  MONITOR = 'monitor'
  GRAPH = 'graph'


class SaveType:
  PICKLE = 'pickle'
  YAML = 'yaml'


class Status:
  TRAINING = 'training'
  SCORE_MET = 'score_met'
  TIMEOUT = 'timeout'
  TARGET_SHIFT = 'target_shift'


class ScoreMetrics:
  SCORE = 'score'
  WIN_RATE = 'win_rate'


class ModelType:
  FORMER = 'former'
  ACTIVE = 'active'
  HARD = 'hard'
  Target = 'target'
