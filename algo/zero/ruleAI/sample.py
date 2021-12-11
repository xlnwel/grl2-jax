import argparse
import collections
import logging
import random
import time
from pathlib import Path
from typing import Union
import numpy as np

from algo.gd.ruleAI.reyn_ai import ReynAIAgent
from core.log import do_logging
from core.elements.agent import Agent
from env.guandan.action import get_action_card, get_action_type
from env.guandan.game import Game
from env.guandan.infoset import get_obs
from env.guandan.utils import PASS, ActionType2Num
from env.typing import EnvOutput
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

    def check_traj_len(self):
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


def run_episode(
        env: Game, 
        agent02: Union[Agent, ReynAIAgent], 
        agent13: ReynAIAgent=ReynAIAgent(), 
        buffers=None,
        test=False):
    env.reset()
    env.start()
    infoset = env.get_infoset()
    obs = get_obs(infoset)
    step = 0
    start = time.time()
    players_traj_lens = [0 for _ in range(4)]
    if test:
        for k, v in obs.items():
            print('\t', k, v)
    while not env.game_over():
        pid = obs['pid']
        if isinstance(agent02, Agent) and (pid == 0 or pid == 2):
            batch_obs = obs.copy()
            batch_obs['eid'] = 0
            for k, v in batch_obs.items():
                batch_obs[k] = np.expand_dims(v, 0)
            reward = np.expand_dims(0, 0)
            discount = np.expand_dims(1, 0)
            reset = np.expand_dims(players_traj_lens[pid] == 0, 0)
            env_output = EnvOutput(batch_obs, reward, discount, reset)
            (action_type, card_rank), _ = agent02(env_output)
            action_type = action_type[0]
            card_rank = card_rank[0]
            assert obs['card_rank_mask'][action_type][card_rank], f"{obs['card_rank_mask']}\n{action_type}\n{card_rank}"
            if action_type == ActionType2Num[PASS]:
                card_rank = 0
            action_id = infoset.action2id(action_type, card_rank)
        else:
            action_id = agent13(infoset)
            action = infoset.legal_actions[action_id]
            action_type = np.int32(get_action_type(action, one_hot=False))
            assert obs['action_type_mask'][action_type] == 1, (action_type, obs['action_type_mask'][action_type])
            card_rank = np.int32(get_action_card(action, one_hot=False))
        card_rank_mask = obs['card_rank_mask']
        assert card_rank_mask[action_type, card_rank] == 1, (action, action_type, card_rank, card_rank_mask)
        env.play(action_id)
        if buffers is not None:
            buffers[pid].add(
                **obs, 
                action_type=action_type, 
                card_rank=card_rank, 
                discount=env.game_over() == False
            )
        infoset = env.get_infoset()
        obs = get_obs(infoset)
        step += 1
        players_traj_lens[pid] += 1
    if test:
        print('end\n\n\n')
        print(list(obs))
        for k, v in obs.items():
            print('\t', k, v)
    max_player_traj_len = max(players_traj_lens)
    scores = env.compute_reward()
    if buffers is not None:
        episodes = [b.sample() for b in buffers]
        episodes = batch_dicts(episodes)
        episodes['reward'] = np.expand_dims(scores, -1) * np.ones_like(episodes['pid'], dtype=np.float32)
        if test:
            print('Episodic stats\n\n\n')
            for k, v in episodes.items():
                print(k, v.shape, v.dtype, v.max(), v.min(), v.mean())
                if len(v.shape) == 2:
                    print('\t', k, '\n', v[:, :2], v[:, -2:])
                elif k == 'rank':
                    v = np.argmax(v, -1)
                    print('\t', k, '\n', v[:, :2], v[:, -2:])
    
        return max_player_traj_len, step, time.time() - start, scores, episodes
    else:
        return max_player_traj_len, step, time.time() - start, scores


def generate_expert_data(n_trajs, log_dir, start_idx=0, test=False):
    agent = ReynAIAgent()
    env = Game(skip_players02=False, skip_players13=False)
    
    buffers = [Buffer(pid=i) for i in range(4)]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    start_idx *= n_trajs
    i = 0
    max_player_traj_len = 0
    while i < n_trajs:
        mptl, step, run_time, scores, episodes = run_episode(env, agent, agent, buffers, test)
        if not np.all([b.check_traj_len() for b in buffers]):
            continue
        filename = log_dir / f'{start_idx+i}-{step}-{int(scores[0])}-{int(scores[1])}-{int(scores[2])}-{int(scores[3])}.npz'
        print(f'{filename}, steps({step}), max player trajectory length({mptl}), duration({run_time})')
        if not test:
            save_data(filename, episodes)
        
        max_player_traj_len = max(max_player_traj_len, mptl)
        steps.append(step)
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
