import uuid
from datetime import datetime
from pathlib import Path
import random
import numpy as np

from core.decorator import config

class EpisodicReplay:
    @config
    def __init__(self):
        self._dir = Path(self._dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory = {}
        self._length = getattr(self, '_length', None)
    
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
            if self._length and len(next(iter(eps.values()))) < self._length + 1:
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

    def sample(self):
        """ Different from other replays, here we only sample a single sequence
        each time as there is little parallel we can do"""
        filename = random.choice(list(self._memory))
        episode = self._memory[filename]
        if self._length:
            total = len(next(iter(episode.values())))
            available = total - self._length
            if available < 1:
                print(f'Skipped short episode of length {available}.')
            index = int(random.randint(0, available))
            episode = {k: v[index: index + self._length] for k, v in episode.items()}
        return episode

    def _remove_files(self):
        if self._max_episodes > 0 and len(self._memory) > self._max_episodes:
            # remove some oldest files if the number of files stored exceeds maximum size
            filenames = sorted(self._memory)
            start = int(.1 * self._max_episodes)
            for filename in filenames[:start]:
                filename.unlink()
                if filename in self._memory:
                    del self._memory[filename]
            filenames = filenames[start:]
            print(f'{start} files are removed')

if __name__ == '__main__':
    import shutil
    directory = 'test_file'
    bs = 10
    sq = 20
    state_shape = (2, )
    replay = EpisodicReplay(directory, bs)
    episodes = [
        dict(
            state=np.random.normal(size=(sq, *state_shape)),
            reward=np.random.normal(size=(sq)),
        )
        for _ in range(bs)
    ]

    replay.merge(episodes)
    directory = Path(directory)
    retrieved_episodes = {}
    filenames = directory.glob('*npz')
    for filename in filenames:
        with filename.open('rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        retrieved_episodes[filename] = episode
        for k in retrieved_episodes[filename].keys():
            try:
                np.testing.assert_allclose(replay._memory[filename][k], retrieved_episodes[filename][k])
            except:
                shutil.rmtree(directory)
    
    replay._memory.clear()
    
    for i in range(bs):
        fn, eps = next(replay.sample())
        for k in eps.keys():
            try:
                np.testing.assert_allclose(retrieved_episodes[fn][k], eps[k])
            except:
                shutil.rmtree(directory)

    shutil.rmtree(directory)