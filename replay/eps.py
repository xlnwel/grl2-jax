import collections
from datetime import datetime
import logging
from pathlib import Path
import random
import uuid
from typing import List
import numpy as np

from core.elements.buffer import Buffer
from core.elements.model import Model
from core.log import do_logging
from core.typing import AttrDict, tree_slice
from replay.local import EpisodicBuffer
from replay.utils import load_data, save_data
from tools.utils import batch_dicts
from tools.display import print_dict_info
from replay import replay_registry

logger = logging.getLogger(__name__)


@replay_registry.register('eps')
class EpisodicReplay(Buffer):
    def __init__(
        self, 
        config: AttrDict, 
        env_stats: AttrDict, 
        model: Model, 
        aid: int=0, 
    ):
        super().__init__(config, env_stats, model, aid)

        self.n_runners = self.config.n_runners
        self.n_envs = self.n_runners * self.config.n_envs

        directory = '/'.join([self.config.root_dir, self.config.model_name])
        self._dir = Path(directory).expanduser()
        self._save = self.config.save
        if self._save:
            self._dir.mkdir(parents=True, exist_ok=True)

        self._filenames = collections.deque()
        self._memory = {}

        self.max_episodes = self.config.get('max_episodes', 1000)
        self.min_episodes = self.config.get('min_episodes', 10)
        self.batch_size = self.config.batch_size
        self.n_recency = self.config.get('n_recency', self.min_episodes)

        self._tmp_bufs: List[EpisodicBuffer] = [
            EpisodicBuffer(config, env_stats, model, aid, 0) 
            for _ in range(self.n_envs)
        ]

    def ready_to_sample(self):
        return len(self) >= self.min_episodes

    def __len__(self):
        return len(self._filenames)

    def add(self, idxes=None, **data):
        if self.n_envs > 1:
            if idxes is None:
                idxes = range(self.n_envs)
            for i in idxes:
                d = tree_slice(data, i)
                eps = self._tmp_bufs[i].add(**d)
                if eps is not None:
                    self.merge(eps)
        else:
            data = tree_slice(data, 0)
            eps = self._tmp_bufs[0].add(**data)
            if eps is not None:
                self.merge(eps)

    def reset_local_buffer(self, i=None):
        if i is None:
            [buf.reset() for buf in self._tmp_bufs]
        elif isinstance(i, (list, tuple)):
            [self._tmp_bufs[ii].reset() for ii in i]
        elif isinstance(i, int):
            self._tmp_bufs[i].reset()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')

    def is_local_buffer_full(self, i=None):
        """ Returns if all local buffers are full """
        if i is None:
            is_full = np.all([buf.is_full() for buf in self._tmp_bufs])
        elif isinstance(i, (list, tuple)):
            is_full = np.all([self._tmp_bufs[ii].is_full() for ii in i])
        elif isinstance(i, int):
            is_full = self._tmp_bufs[i].is_full()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        return is_full

    def finish_episodes(self, i=None):
        """ Adds episodes in local buffers to memory """
        if i is None:
            episodes = [buf.retrieve_all_data() for buf in self._tmp_bufs]
            [buf.reset() for buf in self._tmp_bufs]
        elif isinstance(i, (list, tuple)):
            episodes = [self._tmp_bufs[ii].retrieve_all_data() for ii in i]
            [self._tmp_bufs[ii].reset() for ii in i]
        elif isinstance(i, int):
            episodes = self._tmp_bufs[i].retrieve_all_data()
            self._tmp_bufs[i].reset()
        else:
            raise ValueError(f'{i} of type {type(i)} is not supported')
        self.merge(episodes)
        
    def merge(self, episodes):
        if episodes is None:
            return
        if isinstance(episodes, dict):
            episodes = [episodes]
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        for eps in episodes:
            epslen = len(next(iter(eps.values())))
            if eps is None or (self.sample_size and epslen < self.sample_size):
                # do_logging(f'Ignore short episode of length {epslen}. Minimum acceptable episode length: {self.sample_size}')
                continue    # ignore None/short episodes
            identifier = str(uuid.uuid4().hex)
            filename = self._dir / f'{timestamp}-{identifier}-{epslen}.npz'
            self._memory[filename] = eps
            if self._save:
                save_data(filename, eps)
            self._filenames.append(filename)
        if self._save:
            self._remove_file()
        else:
            self._pop_episode()

    def count_episodes(self):
        """ count the total number of episodes and transitions in the directory """
        if self._save:
            filenames = self._dir.glob('*.npz')
            # subtract 1 as we don't take into account the terminal state
            lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
            episodes, steps = len(lengths), sum(lengths)
            return episodes, steps
        else:
            return 0, 0
    
    def count_steps(self):
        filenames = self._dir.glob('*.npz')
        # subtract 1 as we don't take into account the terminal state
        lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
        episodes, steps = len(lengths), sum(lengths)
        return episodes, steps

    def load_data(self):
        if self._memory == {}:
            # load data from files
            for filename in self._dir.glob('*.npz'):
                if filename not in self._memory:
                    data = load_data(filename)
                    if data is not None:
                        self._filenames.append(filename)
                        self._memory[filename] = data
            do_logging(f'{len(self)} episodes are loaded', logger=logger)
        else:
            do_logging(f'There are already {len(self)} episodes in the memory. No further loading is performed', logger=logger)

    def sample(
        self, 
        batch_size=None, 
        sample_keys=None, 
        sample_size=None, 
        squeeze=False
    ):
        if self.ready_to_sample():
            batch_size = batch_size or self.batch_size
            samples = [self._sample(sample_keys, sample_size, squeeze)
                for _ in range(batch_size)]
            data = batch_dicts(samples)
        else:
            data = None

        return data
    
    def sample_from_recency(
        self, 
        batch_size=None, 
        sample_keys=None, 
        sample_size=1, 
        squeeze=True, 
        n=None
    ):
        """ Sample from the most n recent trajectories. 
        """
        batch_size = batch_size or self.batch_size
        n = n or self.n_recency
        samples = batch_dicts(
            [self._sample(sample_keys, sample_size, squeeze, n) 
            for _ in range(batch_size)]
        )

        return samples

    def _sample(self, sample_keys=None, sample_size=None, squeeze=False, n=None):
        """ Samples a sequence """
        assert set(self._filenames) == set(self._memory), (set(self._memory) - set(self._filenames))
        sample_keys = sample_keys or self.sample_keys
        sample_size = sample_size or self.sample_size
        n = n or len(self._filenames)
        if n is None:
            filename = random.choice(self._filenames)
        else:
            filename = random.choice(list(self._filenames)[-n:])
        episode = self._memory[filename]
        epslen = len(next(iter(episode.values())))
        available = epslen - self.sample_size
        assert available >= 0, f'Skipped short episode of length {epslen}.' \
            f'{[(k, np.array(v).shape) for e in self._memory.values() for k, v in e.items()]}'

        i = random.randint(0, available)
        if sample_size == 1 and squeeze:
            sample = episode.subdict(*sample_keys).slice(i)
        else:
            sample = episode.subdict(*sample_keys).slice(
                np.arange(i, i+sample_size))

        return sample

    def _pop_episode(self):
        if len(self._memory) > self.max_episodes:
            filename = self._filenames.popleft()
            assert(filename in self._memory)
            del self._memory[filename]

    def _remove_file(self):
        if len(self._memory) > self.max_episodes:
            filename = self._filenames.popleft()
            assert(filename in self._memory)
            del self._memory[filename]
            filename.unlink()
            
    def clear_temp_bufs(self):
        for b in self._tmp_bufs:
            b.reset()
