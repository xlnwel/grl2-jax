import argparse
import collections
import logging
import random
import time
from pathlib import Path
import numpy as np

from algo.gd.ruleAI.reyn_ai import ReynAIAgent
from core.log import do_logging
from env.guandan.action import get_action_card, get_action_type
from env.guandan.game import Game
from env.guandan.infoset import get_obs
from replay.utils import save_data, load_data
from utility.utils import batch_dicts

logger = logging.getLogger(__name__)


class Buffer:
    def __init__(self, pid):
        self._pid = pid
        self._memory = collections.defaultdict(list)
        self._idx = 0

    def traj_len(self):
        return self._idx

    def reset(self):
        self._memory.clear()
        self._idx = 0

    def check(self):
        return self._idx <= 40

    def sample(self):
        data = {k: np.array(v) for k, v in self._memory.items()}
        results = {k: np.zeros((40, *v.shape[1:]), dtype=v.dtype) 
            for k, v in data.items()}
        for k, v in data.items():
            results[k][:self._idx] = v
        self.reset()
        return results

    def add(self, **data):
        assert data['pid'] == self._pid, data['pid']
        for k, v in data.items():
            self._memory[k].append(v)
        self._memory['mask'].append(np.float32(1))
        self._idx += 1


def generate_expert_data(n_trajs, log_dir, start_idx=0, test=False):
    agent = ReynAIAgent()
    env = Game()
    
    buffers = [Buffer(pid=i) for i in range(4)]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    start_idx *= n_trajs
    i = 0
    max_player_traj_len = 0
    while i < n_trajs:
        env.reset()
        env.start()
        infoset = env.get_infoset()
        obs = get_obs(infoset)
        step = 0
        start = time.time()
        if test:
            for k, v in obs.items():
                print('\t', k, v)
        while not env.game_over():
            # for k, v in obs.items():
            #     if isinstance(v, np.ndarray):
            #         print(k, v.shape, v.dtype, v.max(), v.min())
            action_id = agent(infoset)
            action = infoset.legal_actions[action_id]
            action_type = np.int32(get_action_type(action, one_hot=False))
            assert obs['action_type_mask'][action_type] == 1, obs['action_type_mask'][action_type]
            card_rank = np.int32(get_action_card(action, one_hot=False))
            if action_type != 0:
                card_rank_mask = obs['follow_mask'] if action_type == 1 else obs['bomb_mask']
                assert card_rank_mask[card_rank] == 1, (action, action_type, card_rank, card_rank_mask)
            env.play(action_id)
            buffers[obs['pid']].add(
                **obs, 
                action_type=action_type, 
                card_rank=card_rank, 
                discount=env.game_over() == False
            )
            infoset = env.get_infoset()
            obs = get_obs(infoset)
            step += 1
        if test:
            print('end\n\n\n')
            print(list(obs))
            for k, v in obs.items():
                print('\t', k, v)

        max_player_traj_len = max([max_player_traj_len, *[b.traj_len() for b in buffers]])
        if not np.all([b.check() for b in buffers]):
            continue
        steps.append(step)
        rewards = env.compute_reward()
        episodes = [b.sample() for b in buffers]
        episodes = batch_dicts(episodes)
        episodes['reward'] = np.expand_dims(rewards, -1) * np.ones_like(episodes['pid'], dtype=np.float32)
        if test:
            print('Episodic stats\n\n\n')
            for k, v in episodes.items():
                print(k, v.shape, v.dtype, v.max(), v.min(), v.mean())
                if len(v.shape) == 2:
                    print('\t', k, '\n', v[:, :2], v[:, -2:])
                elif k == 'rank':
                    v = np.argmax(v, -1)
                    print('\t', k, '\n', v[:, :2], v[:, -2:])
        # for k, v in episode.items():
        #     if isinstance(v, dict):
        #         for kk, vv in v.items():
        #             print(k, kk, vv.shape, vv.dtype, vv.max(), vv.min(), vv.mean())
        #     else:
        #         print(k, v.shape, v.dtype, v.max(), v.min(), v.mean())
        # assert False
        filename = log_dir / f'{start_idx+i}-{step}-{int(rewards[0])}-{int(rewards[1])}-{int(rewards[2])}-{int(rewards[3])}.npz'
        print(f'{filename}, steps({step}), duration({time.time()-start})')
        if not test:
            save_data(filename, episodes)
        i += 1
    
    return max_player_traj_len, steps

def distributedly_generate_expert_data(n_trajs, workers, directory):
    import ray

    ray.init()

    rged = ray.remote(generate_expert_data)
    start = time.time()
    obj_ids = [rged.remote(n_trajs // workers, directory, start_idx=i)
        for i in range(workers)]
    max_player_traj_lens, steps = list(zip(*ray.get(obj_ids)))
    steps = sum(steps, [])
    print('Total time for sampling:', time.time() - start)
    print('Maximum trajectory length:', max(steps))
    print("Maximum player's trajectory length:", max(max_player_traj_lens))

    ray.shutdown()

    return steps


def load_all_data(directory):
    if isinstance(directory, str):
        directory = Path(directory)
    memory = {}
    for filename in directory.glob('*.npz'):
        data = load_data(filename)
        if data is not None:
            memory[filename] = data
    do_logging(f'{len(memory)} episodes are loaded', logger=logger)

    return memory

def read_an_episode(directory):
    directory = Path(directory)
    filename = random.choice(list(directory.glob('*.npz')))
    print(f'filename: {filename}')
    data = load_data(filename)
    if data is not None:
        i = np.random.randint(len(next(iter(data.values()))))
        for k, v in data.items():
            print(k, v.shape, v.dtype, v.max(), v.min(), v.mean())
            # print(len(v), i, v[i])
    else:
        print('No data is loaded')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', '-s', action='store_true')
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--n_trajs', '-n', type=int, default=100000)
    parser.add_argument('--workers', '-w', type=int, default=10)
    parser.add_argument('--directory', '-d', default='gd')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    directory = f'data/{args.directory}-{args.n_trajs}'
    if args.test:
        steps = generate_expert_data(1, directory, test=True)
    elif args.store:
        steps = distributedly_generate_expert_data(args.n_trajs, args.workers, directory)
    else:
        read_an_episode(directory)
