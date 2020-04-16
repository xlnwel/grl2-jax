import uuid
from datetime import datetime
from pathlib import Path
import random
import numpy as np

from core.decorator import config

class EpisodicReplay:
    @config
    def __init__(self, state_keys=[]):
        self._dir = Path(self._dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory = {}
        self._batch_len = getattr(self, '_batch_len', None)
        self._state_keys = state_keys
    
    def buffer_type(self):
        return self._type

    def good_to_learn(self):
        return len(self._memory) >= self._min_episodes

    def __len__(self):
        return len(self._memory)

    def merge(self, episodes):
        if isinstance(episodes, dict):
            episodes = [episodes]
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        for eps in episodes:
            if self._batch_len and len(next(iter(eps.values()))) < self._batch_len + 1:
                continue
            identifier = str(uuid.uuid4().hex)
            length = len(eps['reward'])
            filename = self._dir / f'{timestamp}-{identifier}-{length}.npz'
            self._memory[filename] = eps
            with filename.open('wb') as f1:
                np.savez_compressed(f1, **eps)
        self._remove_files()

    def count_episodes(self):
        """ count the total number of episodes and transitions in the directory """
        filenames = self._dir.glob('*.npz')
        # subtract 1 as we don't take into account the terminal state
        lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
        episodes, steps = len(lengths), sum(lengths)
        return episodes, steps
    
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
                    try:
                        with filename.open('rb') as f:
                            episode = np.load(f)
                            episode = {k: episode[k] for k in episode.keys()}
                    except Exception as e:
                        print(f'Could not load episode: {e}')
                        continue
                    self._memory[filename] = episode
            print(f'{len(self._memory)} episodes are loaded')

    def sample(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        if batch_size > 1:
            samples = [self._sample() for _ in range(batch_size)]
            data = {k: np.stack([t[k] for t in samples], 0)
                for k in samples[0].keys()}
        else:
            data = self._sample()
        return data

    def _sample(self):
        filename = random.choice(list(self._memory))
        episode = self._memory[filename]
        if self._batch_len:
            total = len(next(iter(episode.values())))
            available = total - self._batch_len
            if available < 1:
                print(f'Skipped short episode of length {available}.')
            i = int(random.randint(0, available))
            episode = {k: v[i] if k in self._state_keys else v[i: i + self._batch_len] 
                        for k, v in episode.items()}
        return episode

    def _remove_files(self):
        if getattr(self, '_max_episodes', 0) > 0 and len(self._memory) > self._max_episodes:
            # remove some oldest files if the number of files stored exceeds maximum size
            filenames = sorted(self._memory)
            start = int(.1 * self._max_episodes)
            for filename in filenames[:start]:
                filename.unlink()
                if filename in self._memory:
                    del self._memory[filename]
            filenames = filenames[start:]
            print(f'{start} files are removed')