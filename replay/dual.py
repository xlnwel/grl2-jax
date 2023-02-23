import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.typing import AttrDict
from tools.utils import batch_dicts
from replay import replay_registry


FAST_REPLAY = 'fast'
SLOW_REPLAY = 'slow'


@replay_registry.register('dual')
class DualReplay(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.n_envs = self.config.n_runners * self.config.n_envs

        self._batch_size = config.batch_size
        self._fast_percentage = config.fast_percentage
        fast_config = config.fast_replay
        slow_config = config.slow_replay
        fast_config.batch_size = int(self._batch_size * self._fast_percentage)
        slow_config.batch_size = int(self._batch_size * (1 - self._fast_percentage))
        assert fast_config.batch_size + slow_config.batch_size == self._batch_size, f'batch size({self._batch_size}) can not be partitioned by fast_percentation({self._fast_percentage})'

        if config.recent_fast_replay:
            fast_config.max_size = self.n_envs * config.n_steps
            fast_config.min_size = fast_config.batch_size
        self._fast_config = fast_config
        self._slow_config = slow_config
        self._fast_type = self._fast_config.type
        self._slow_type = self._slow_config.type

        self.fast_replay = self.build_replay(self._fast_config, model)
        self.slow_replay = self.build_replay(self._slow_config, model)

        self.default_replay = FAST_REPLAY

    @property
    def fast_config(self):
        return self._fast_config

    @property
    def slow_config(self):
        return self._slow_config
    
    @property
    def fast_type(self):
        return self._fast_type

    @property
    def slow_type(self):
        return self._slow_type

    def build_replay(self, config: AttrDict, model: Model):
        Cls = replay_registry.get(config.type)
        replay = Cls(config, self.env_stats, model, self.aid)
        return replay
    
    def set_default_replay(self, target_replay):
        assert target_replay in [FAST_REPLAY, SLOW_REPLAY], target_replay
        self.default_replay = target_replay
        
    def ready_to_sample(self):
        return self.slow_replay.ready_to_sample() and self.fast_replay.ready_to_sample()

    def __len__(self):
        return len(self.slow_replay) + len(self.fast_replay)

    def collect(self, target_replay=None, **data):
        if target_replay is None:
            target_replay = self.default_replay
        if target_replay == FAST_REPLAY:
            popped_data = self.fast_replay.collect_and_pop(**data)
            self.slow_replay.merge(popped_data)
        elif target_replay == SLOW_REPLAY:
            self.slow_replay.collect(**data)
        else:
            raise NotImplementedError(target_replay)

    def sample(self, batch_size=None):
        if self.ready_to_sample():
            return self._sample(batch_size)
        else:
            return None

    def merge(self, local_buffer, target_replay=None):
        if target_replay is None:
            target_replay = self.default_replay
        if target_replay == FAST_REPLAY:
            popped_data = self.fast_replay.merge_and_pop(local_buffer)
            self.slow_replay.merge(popped_data)
        elif target_replay == SLOW_REPLAY:
            self.slow_replay.merge(local_buffer)
        else:
            raise NotImplementedError(target_replay)
    
    """ Implementation """
    def _sample(self, batch_size=None):
        if batch_size is not None:
            assert int(batch_size * self._fast_percentage) == batch_size * self._fast_percentage
            fast_bs = int(batch_size * self._fast_percentage)
            slow_bs = int(batch_size * (1 - self._fast_percentage))
        else:
            fast_bs, slow_bs = None, None
        
        fast_data = self.fast_replay.sample(fast_bs)
        slow_data = self.slow_replay.sample(slow_bs)
        data = batch_dicts([fast_data, slow_data], np.concatenate)
        
        return data
        
    def _move_data_from_fast_to_slow(self):
        data = self.fast_replay.retrive_all_data()
        self.slow_replay.merge(data)
