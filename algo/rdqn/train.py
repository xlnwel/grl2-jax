import time
import functools
import numpy as np
import tensorflow as tf

from core.tf_config import *
from utility.utils import Every, TempStore
from utility.graph import video_summary
from utility.timer import TBTimer
from utility.run import Runner, evaluate
from utility import pkg
from env.gym_env import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env


def train(agent, env, eval_env, replay):
    def collect(env, step, reset, obs, action, reward, next_obs, **kwargs):
        kwargs['obs'] = env.prev_obs() if reset else next_obs
        kwargs['prev_action'] = action
        kwargs['prev_reward'] = reward
        replay.add(**kwargs)
        if reset:
            replay.clear_temp_buffer()
            replay.pre_add(obs=next_obs)
    
    step = agent.env_step
    runner = Runner(env, agent, step=step)
    while not replay.good_to_learn():
        step = runner.run(step_fn=collect, nsteps=int(1e4))

    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step < int(agent.MAX_STEPS):
        start_step = step
        start_t = time.time()
        agent.learn_log(step)
        step = runner.run(step_fn=collect, nsteps=agent.TRAIN_PERIOD)
        duration = time.time() - start_t
        agent.store(
            fps=(step-start_step) / duration,
            tps=(agent.N_UPDATES / duration))

        if to_eval(step):
            with TempStore(agent.get_states, agent.reset_states):
                eval_score, eval_epslen, video = evaluate(
                    eval_env, agent, record=False, size=(64, 64))
                # video_summary(f'{agent.name}/sim', video, step=step)
                agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        
        if to_log(step):
            agent.log(step)
            agent.save()
    

def main(env_config, model_config, agent_config, replay_config):
    algo = agent_config['algorithm']
    env = env_config['name']
    if 'atari' not in env:
        from utility import yaml_op
        root_dir = agent_config['root_dir']
        model_name = agent_config['model_name']
        directory = pkg.get_package(algo, 0, '/')
        config = yaml_op.load_config(f'{directory}/config2.yaml')
        env_config = config['env']
        model_config = config['model']
        agent_config = config['agent']
        replay_config = config['replay']
        agent_config['root_dir'] = root_dir
        agent_config['model_name'] = model_name
        env_config['name'] = env

    create_model, Agent = pkg.import_agent(config=agent_config)

    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    assert env.n_envs == 1, \
        f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
    eval_env_config = env_config.copy()
    eval_env_config.pop('reward_clip', False)
    eval_env_config.pop('life_done', False)
    eval_env = create_env(eval_env_config)

    models = create_model(model_config, env)

    replay_config['dir'] = agent_config['root_dir'].replace('logs', 'data')
    replay = create_replay(replay_config, state_keys=['h', 'c'], prev_action=0, prev_reward=0)
    data_format = pkg.import_module('agent', algo).get_data_format(
        env, agent_config['batch_size'], agent_config['sample_size'],
        replay_config['replay_type'].endswith('per'), agent_config['store_state'], 
        models['q'].state_size)
    
    process = functools.partial(process_with_env, 
        env=env, obs_range=[0, 1], one_hot_action=False)
    dataset = Dataset(replay, data_format, process_fn=process)

    # construct agent
    agent = Agent(
        name='q',
        config=agent_config, 
        models=models, 
        dataset=dataset, 
        env=env)
    
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))

    train(agent, env, eval_env, replay)
