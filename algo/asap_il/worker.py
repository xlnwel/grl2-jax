import random
import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import TBTimer
from utility.utils import step_str
from env.gym_env import create_gym_env
from algo.apex.buffer import create_local_buffer
from algo.asap.worker import Worker
from algo.asap.utils import *


class ILWorker(Worker):
    def _collect_data(self, buffer, store_data, tag, action_std, step, **kwargs):
        if store_data:
            buffer.add_data(kl_flag=tag==Tag.EVOLVED, **kwargs)
        if np.any(action_std != 0):
            self.store(**{f'{tag}_action_std': np.mean(action_std)})
        self._periodic_logging(step)
