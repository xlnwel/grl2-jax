import logging
import numpy as np

from core.typing import dict2AttrDict
from tools.utils import batch_dicts
from tools.display import print_dict_info
from algo.ppo.elements.buffer import LocalBuffer, ACBuffer as BufferBase

logger = logging.getLogger(__name__)


def concat_obs(obs, last_obs):
    obs = np.concatenate([obs, np.expand_dims(last_obs, 0)], 0)
    return obs



class ACBuffer(BufferBase):
    def _sample(self, sample_keys=None):
        sample_keys = sample_keys or self.sample_keys
        samples = self._queue
        self._queue = []
        sample = batch_dicts(samples, func=np.concatenate)

        return sample


def create_buffer(config, model, env_stats, **kwargs):
    config = dict2AttrDict(config)
    env_stats = dict2AttrDict(env_stats)
    BufferCls = {
        'ac': ACBuffer, 
        'local': LocalBuffer
    }[config.type]
    return BufferCls(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
