import uuid
from datetime import datetime
from pathlib import Path
import numpy as np


class EpisodicReplay:
    def __init__(self, directory, rescan, length=None):
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory = {}
    
    def merge(self, episodes):
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        for eps in episodes:
            identifier = str(uuid.uuid4().hex)
            length = len(eps['reward'])
            filename = self._dir / f'{timestamp}-{identifier}-{length}.npz'
            self._memory[filename] = eps
            with filename.open('wb') as f1:
                np.savez_compressed(f1, **eps)

    def count_episodes(self):
        """ count the total number of episodes and transitions in the directory """
        filenames = self._dir.glob('*.npz')
        lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
        episodes, steps = len(lengths), sum(lengths)
        return episodes, steps

    def sample(self, length=None, balance=False, seed=0):
        """ Different from other replays, here we only sample a single sequence
        each time as there is little parallel we can do"""
        random = np.random.RandomState(seed)
        while True:
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
            keys = list(self._memory.keys())
            for index in random.choice(len(keys), self._rescan):
                episode = self._memory[keys[index]]
                if length:
                    total = len(next(iter(episode.values())))
                    available = total - length
                    if available < 1:
                        print(f'Skipped short episode of length {available}.')
                        continue
                    if balance:
                        index = min(random.randint(0, total), available)
                    else:
                        index = int(random.randint(0, available))
                    episode = {k: v[index: index + length] for k, v in episode.items()}
                yield episode
