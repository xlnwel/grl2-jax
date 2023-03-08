import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.typing import AttrDict
from tools.utils import batch_dicts
from replay import replay_registry


PRIMAL_REPLAY = 'primal'
SECONDARY_REPLAY = 'secondary'


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

        self._detached =  config.detached
        self._batch_size = config.batch_size
        self._primal_percentage = config.primal_percentage
        primal_config = config.primal_replay
        secondary_config = config.secondary_replay
        primal_config.batch_size = int(self._batch_size * self._primal_percentage)
        secondary_config.batch_size = int(self._batch_size * (1 - self._primal_percentage))
        assert primal_config.batch_size + secondary_config.batch_size == self._batch_size, f'batch size({self._batch_size}) can not be partitioned by fast_percentation({self._primal_percentage})'

        if config.recent_primal_replay:
            primal_config.max_size = self.n_envs * config.n_steps
            primal_config.min_size = primal_config.batch_size
        self._primal_config = primal_config
        self._secondary_config = secondary_config
        self._primal_type = self._primal_config.type
        self._secondary_type = self._secondary_config.type

        self.primal_replay = self.build_replay(self._primal_config, model)
        self.secondary_replay = self.build_replay(self._secondary_config, model)

        self.default_replay = PRIMAL_REPLAY

    @property
    def primal_config(self):
        return self._primal_config

    @property
    def secondary_config(self):
        return self._secondary_config
    
    @property
    def primal_type(self):
        return self._primal_type

    @property
    def secondary_type(self):
        return self._secondary_type

    def build_replay(self, config: AttrDict, model: Model):
        Cls = replay_registry.get(config.type)
        replay = Cls(config, self.env_stats, model, self.aid)
        return replay
    
    def set_default_replay(self, target_replay=PRIMAL_REPLAY):
        assert target_replay in [PRIMAL_REPLAY, SECONDARY_REPLAY], target_replay
        self.default_replay = target_replay
        
    def ready_to_sample(self, target_replay=None):
        if target_replay == PRIMAL_REPLAY:
            return self.primal_replay.ready_to_sample()
        elif target_replay == SECONDARY_REPLAY:
            return self.secondary_replay.ready_to_sample()
        else:
            return self.secondary_replay.ready_to_sample() and self.primal_replay.ready_to_sample()

    def __len__(self):
        return len(self.secondary_replay) + len(self.primal_replay)

    def collect(self, target_replay=None, **data):
        if target_replay is None:
            target_replay = self.default_replay
        if target_replay == PRIMAL_REPLAY:
            if self._detached:
                self.primal_replay.collect(**data)
            else:
                popped_data = self.primal_replay.collect_and_pop(**data)
                self.secondary_replay.merge(popped_data)
        elif target_replay == SECONDARY_REPLAY:
            self.secondary_replay.collect(**data)
        else:
            raise NotImplementedError(target_replay)

    def merge(self, local_buffer, target_replay=None):
        if target_replay is None:
            target_replay = self.default_replay
        if target_replay == PRIMAL_REPLAY:
            if self._detached:
                self.primal_replay.merge(local_buffer)
            else:
                popped_data = self.primal_replay.merge_and_pop(local_buffer)
                self.secondary_replay.merge(popped_data)
        elif target_replay == SECONDARY_REPLAY:
            self.secondary_replay.merge(local_buffer)
        else:
            raise NotImplementedError(target_replay)

    def sample(self, batch_size=None, primal_percentage=None):
        if not self.ready_to_sample(SECONDARY_REPLAY):
            primal_percentage = 1
        if primal_percentage == 1:
            return self.primal_replay.sample(self._batch_size)
        elif primal_percentage == 0:
            return self.secondary_replay.sample(self._batch_size)
        else:
            target_replay = None
        if self.ready_to_sample(target_replay):
            return self._sample(batch_size, primal_percentage=primal_percentage)
        else:
            return None

    def sample_from_recency(self, target_replay=PRIMAL_REPLAY, **kwargs):
        if target_replay == PRIMAL_REPLAY:
            return self.primal_replay.sample_from_recency(**kwargs)
        elif target_replay == SECONDARY_REPLAY:
            return self.secondary_replay.sample_from_recency(**kwargs)
        else:
            raise NotImplementedError(target_replay)

    def ergodic_sample(self, target_replay=PRIMAL_REPLAY, batch_size=None):
        if target_replay == PRIMAL_REPLAY:
            return self.primal_replay.ergodic_sample(batch_size)
        elif target_replay == SECONDARY_REPLAY:
            return self.secondary_replay.ergodic_sample(batch_size)
        else:
            raise NotImplementedError(target_replay)

    def get_obs_rms(self, target_replay=PRIMAL_REPLAY):
        if target_replay == PRIMAL_REPLAY:
            return self.primal_replay.get_obs_rms()
        elif target_replay == SECONDARY_REPLAY:
            return self.secondary_replay.get_obs_rms()
        else:
            raise NotImplementedError(target_replay)

    def clear_local_buffer(self, target_replay=None):
        if target_replay is None:
            target_replay = self.default_replay
        if target_replay == PRIMAL_REPLAY:
            self.primal_replay.clear_local_buffer()
        if target_replay == SECONDARY_REPLAY:
            self.secondary_replay.clear_local_buffer()
        else:
            raise NotImplementedError(target_replay)

    """ Implementation """
    def _sample(self, batch_size=None, primal_percentage=None):
        if primal_percentage is None:
            primal_percentage = self._primal_percentage
        if batch_size is None:
            batch_size = self._batch_size
        fast_bs = int(batch_size * primal_percentage)
        slow_bs = batch_size - fast_bs

        if primal_percentage == 1:
            data = self.primal_replay.sample(fast_bs)
        elif primal_percentage == 0:
            data = self.secondary_replay.sample(slow_bs)
        else:
            primal_data = self.primal_replay.sample(fast_bs)
            secondary_data = self.secondary_replay.sample(slow_bs)
            data = batch_dicts([primal_data, secondary_data], np.concatenate)
        
        return data
        
    def _move_data_from_fast_to_slow(self):
        data = self.primal_replay.retrive_all_data()
        self.secondary_replay.merge(data)
