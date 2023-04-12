import collections
from typing import List
import numpy as np

from core.ckpt.pickle import save, restore
from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from core.typing import AttrDict, subdict
from replay.local import NStepBuffer
from tools.utils import batch_dicts, yield_from_tree
from tools.rms import RunningMeanStd
from tools.timer import Timer
from replay import replay_registry


@replay_registry.register('uniform')
class UniformReplay(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.n_envs = self.config.n_runners * self.config.n_envs

        self.max_size = int(self.config.max_size)
        self.min_size = int(self.config.min_size)
        self.batch_size = self.config.batch_size
        self.n_recency = self.config.get('n_recency', self.min_size)
        self.n_steps = self.config.n_steps

        if self.config.directory:
            filedir = self.config.directory
        elif self.config.root_dir and self.config.model_name:
            filedir = '/'.join([self.config.root_dir, self.config.model_name])
        else:
            filedir = None
        self._filedir = filedir
        self._filename = self.config.filename if self.config.filename else 'uniform'

        self._memory = collections.deque(maxlen=self.max_size)

        if self.config.model_norm_obs:
            self.obs_rms = RunningMeanStd([0])
        self._tmp_bufs: List[NStepBuffer] = [
            NStepBuffer(config, env_stats, model, aid, 0) 
            for _ in range(self.n_envs)
        ]

    def __len__(self):
        return len(self._memory)

    def ready_to_sample(self):
        return len(self._memory) >= self.min_size

    def reset(self):
        pass
    
    def collect_and_pop(self, idxes=None, **data):
        data = self._prepare_data(**data)
        return self.add_and_pop(idxes=idxes, **data)

    def add(self, idxes=None, **data):
        trajs = []
        if self.n_envs > 1:
            for i, d in enumerate(yield_from_tree(data)):
                if i >= len(self._tmp_bufs):
                    self._tmp_bufs.append(NStepBuffer(
                        self.config, self.env_stats, self.model, self.aid, 0))
                traj = self._tmp_bufs[i].add(**d)
                if traj is not None:
                    trajs.append(traj)
        else:
            traj = self._tmp_bufs[0].add(**data)
            if traj is not None:
                trajs.append(traj)
        self.merge(trajs)

    def add_and_pop(self, idxes=None, **data):
        trajs = []
        popped_data = []
        if self.n_envs > 1:
            for i, d in enumerate(yield_from_tree(data)):
                if i >= len(self._tmp_bufs):
                    self._tmp_bufs.append(NStepBuffer(
                        self.config, self.env_stats, self.model, self.aid, 0))
                traj = self._tmp_bufs[i].add(**d)
                if traj is not None:
                    trajs.append(traj)
        else:
            traj = self._tmp_bufs[0].add(**data)
            if traj is not None:
                trajs.append(traj)
        popped_data.extend(self.merge_and_pop(trajs))

        return popped_data

    def merge(self, trajs):
        if isinstance(trajs, dict):
            trajs = [trajs]
        self._memory.extend(trajs)
        assert len(self) <= self.max_size, len(self)
        self._update_obs_rms(trajs)

    def merge_and_pop(self, trajs):
        if isinstance(trajs, dict):
            trajs = [trajs]
        popped_data = []
        for traj in trajs:
            if len(self._memory) == self._memory.maxlen:
                popped_data.append(self._memory.popleft())
            self._memory.append(traj)
        self._update_obs_rms(trajs)
        return popped_data

    def sample_from_recency(self, batch_size, sample_keys, n=None, **kwargs):
        batch_size = batch_size or self.batch_size
        n = max(batch_size, n or self.n_recency)
        if n > len(self):
            return None
        idxes = np.arange(len(self)-n, len(self))
        idxes = np.random.choice(idxes, size=batch_size, replace=False)

        samples = batch_dicts([subdict(self._memory[i], *sample_keys) for i in idxes])

        return samples
        
    def sample(self, batch_size=None):
        if self.ready_to_sample():
            samples = self._sample(batch_size)
        else:
            samples = None

        return samples

    def ergodic_sample(self, batch_size=None, n=None, start=None):
        if not self.ready_to_sample():
            return None
        batch_size = batch_size or self.batch_size
        n = len(self) if n is None else min(n, len(self))
        n = n // batch_size * batch_size
        if start is None:
            idxes = np.arange(len(self))[-n:]
        else:
            idxes = np.arange(len(self))[-(start + n):-start]
        np.random.shuffle(idxes)
        # print('ergodic sample idx max', np.max(idxes), 'min', np.min(idxes))
        for i in range(0, n, batch_size):
            yield self._get_samples(idxes[i:i+batch_size], self._memory)
    
    def range_sample(self, start, n):
        if not self.ready_to_sample():
            return None
        end = min(len(self), start+n)
        idxes = np.arange(start, end)
        # print('range sample idx max', np.max(idxes), 'min', np.min(idxes))
        return self._get_samples(idxes, self._memory)

    def get_obs_rms(self):
        if self.config.model_norm_obs:
            rms = self.obs_rms.get_rms_stats() \
                if self.obs_rms.is_initialized() else None
            self.obs_rms.reset_rms_stats()
            assert not self.obs_rms.is_initialized(), (self.obs_rms._count, self.obs_rms._epsilon, self.obs_rms.is_initialized())
            return rms        

    def retrieve_all_data(self):
        self.clear_local_buffer()
        data = self._memory
        self._memory = collections.deque(maxlen=self.max_size)
        return data

    def clear_local_buffer(self, drop_data=False):
        for b in self._tmp_bufs:
            if drop_data:
                b.reset()
            else:
                traj = b.retrieve_all_data()
                if traj:
                    self.merge(traj)

    def _sample(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        idxes = np.random.randint(len(self), size=batch_size)
        # the following code avoids repetitive sampling, 
        # but it takes significant more time to run(around 1000x).
        # idxes = np.random.choice(size, size=batch_size, replace=False)
        
        samples = self._get_samples(idxes, self._memory)

        return samples

    def _get_samples(self, idxes, memory):
        fn = lambda x: np.expand_dims(np.stack(x), 1)
        samples = batch_dicts([memory[i] for i in idxes], func=fn)

        return samples

    def _update_obs_rms(self, trajs):
        if self.config.model_norm_obs:
            if self.config.model_use_state:
                self.obs_rms.update(
                    np.stack([traj['env_state'] for traj in trajs]), 
                )
            else:
                self.obs_rms.update(
                    np.stack([traj['obs'] for traj in trajs]), 
                )

    def save(self, filedir=None, filename=None):
        filedir = filedir or self._filedir
        filename = filename or self._filename
        save(self._memory, filedir=filedir, filename=filename, name='data')
        do_logging(f'Number of transitions saved: {len(self)}')
    
    def restore(self, filedir=None, filename=None):
        filedir = filedir or self._filedir
        filename = filename or self._filename
        self._memory = restore(filedir=filedir, filename=filename, 
            default=collections.deque(maxlen=self.max_size), name='data')
        do_logging(f'Number of transitions restored: {len(self)}')
