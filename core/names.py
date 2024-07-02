import os

ANCILLARY = 'ancillary'
OPTIMIZER = 'opt'
MODEL = 'model'
LOSS = 'loss'
TRAINER = 'trainer'
ACTOR = 'actor'
BUFFER = 'buffer'
STRATEGY = 'strategy'
AGENT = 'agent'

PARAMS = 'params'
OBS = 'obs'
REWARD = 'reward'
DISCOUNT = 'discount'
DEFAULT_ACTION = 'action'

ROOT_DIR = 'root_dir'
MODEL_NAME = 'model_name'

TRAIN_STEP = 'train_step'
ENV_STEP = 'env_step'

if os.name == 'posix':  # Linux, macOS, or Unix
  PATH_SPLIT = '/'
elif os.name == 'nt':  # Windows
  PATH_SPLIT = '\\'
else:
  raise RuntimeError(f'Unsupported system ({os.name})')


class TRAIN_AXIS:
  BATCH = 0
  SEQ = 1
  UNIT = 2

class INFER_AXIS:
  BATCH = 0
  UNIT = 1

class DL_LIB:
  TORCH = 'th'
  JAX = 'jax'
