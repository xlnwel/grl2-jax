import argparse
import logging
import random
import time
from pathlib import Path
import numpy as np

from algo.gd.ruleAI.reyn_ai import ReynAIAgent
from core.log import do_logging
from env.guandan.action import get_action_card, get_action_type
from env.guandan.small_game import SmallGame
from env.guandan.guandan_env import get_obs
from replay.local import EnvEpisodicBuffer
from replay.utils import save_data, load_data

logger = logging.getLogger(__name__)


def generate_expert_data(n_trajs, log_dir, start_idx=0, test=False):
    agent = ReynAIAgent()
    env = SmallGame()
    
    local_buffer= EnvEpisodicBuffer({})
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    start_idx *= n_trajs
    for i in range(n_trajs):
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
            action_type = get_action_type(action)
            action_card = get_action_card(action)
            env.play(action_id)
            local_buffer.add(
                **obs, 
                action_type=action_type, 
                action_card=action_card, 
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
        # assert False
        steps.append(step)
        reward = env.compute_reward()
        episode = local_buffer.sample()
        episode['reward'] = reward[episode['pid']]
        # for k, v in episode.items():
        #     if isinstance(v, dict):
        #         for kk, vv in v.items():
        #             print(k, kk, vv.shape, vv.dtype, vv.max(), vv.min(), vv.mean())
        #     else:
        #         print(k, v.shape, v.dtype, v.max(), v.min(), v.mean())
        # assert False
        filename = log_dir / f'{start_idx+i}-{step}-{int(reward[0])}-{int(reward[1])}-{int(reward[2])}-{int(reward[3])}.npz'
        print(filename, step, time.time()-start)
        if not test:
            save_data(filename, episode)
    
    return steps

def distributedly_generate_expert_data(n_trajs, workers, directory):
    import ray

    ray.init()

    rged = ray.remote(generate_expert_data)
    start = time.time()
    obj_ids = [rged.remote(n_trajs // workers, directory, start_idx=i)
        for i in range(workers)]
    steps = ray.get(obj_ids)
    steps = sum(steps, [])
    print(time.time() - start)

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
            print(k, v.shape, v.max(), v.min(), v.mean())
            # print(len(v), i, v[i])
    else:
        print('No data is loaded')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', '-s', action='store_true')
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--n_trajs', '-n', type=int, default=1000000)
    parser.add_argument('--workers', '-w', type=int, default=10)
    parser.add_argument('--directory', '-d', default='gd')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    directory = f'data/{args.directory}'
    if args.test:
        steps = generate_expert_data(1, directory, test=True)
    elif args.store:
        steps = distributedly_generate_expert_data(args.n_trajs, args.workers, directory)
        print(np.max(steps))
    else:
        read_an_episode(directory)
