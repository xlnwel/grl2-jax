import time
import functools
import numpy as np
import tensorflow as tf

from core.tf_config import *
from utility.utils import Every
from utility.graph import video_summary
from utility.timer import TBTimer
from utility.run import Runner, evaluate
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env


def train(agent, env, eval_env, replay):
    def collect_and_learn(env, step, reset, **kwargs):
        if reset:
            # we reset noisy every episode. Theoretically, 
            # this follows the guide of deep exploration.
            # More importantly, it saves time!
            if hasattr(agent, 'reset_noisy'):
                agent.reset_noisy()
        replay.add(**kwargs)
        if step % agent.TRAIN_PERIOD == 0:
            agent.train_record(step)
    
    step = agent.env_step
    collect = lambda *args, **kwargs: replay.add(**kwargs)
    runner = Runner(env, agent, step=step, nsteps=agent.LOG_PERIOD)
    def random_actor(*args, **kwargs):
        ar = np.random.randint(0, agent.ar.max_ar)
        return env.random_action(), ar, {'ar': ar}
    while not replay.good_to_learn():
        step = runner.run(
            action_selector=random_actor, 
            step_fn=collect)

    to_eval = Every(agent.EVAL_PERIOD)
    print('Training starts...')
    while step <= int(agent.MAX_STEPS):
        start_step = step
        start = time.time()
        step = runner.run(step_fn=collect_and_learn)
        fps = (step - start_step) / (time.time() - start)
        agent.store(
            env_step=agent.env_step,
            train_step=agent.train_step,
            fps=fps, 
            tps=fps/agent.TRAIN_PERIOD)

        if to_eval(step):
            n = 10 if 'procgen' in eval_env.name else 1
            eval_score, eval_epslen, video = evaluate(
                eval_env, agent, record=agent.RECORD, n=n)
            if agent.RECORD:
                video_summary(f'{agent.name}/sim', video, step=step)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        agent.record(step=step)
        agent.save()

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    env = create_env(env_config)
    assert env.n_envs == 1, \
        f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
    # if env_config['name'].startswith('procgen'):
    #     start_level = 200
    eval_env_config = env_config.copy()
    eval_env_config.pop('reward_clip', False)
    eval_env = create_env(eval_env_config)
    replay = create_replay(replay_config)

    am = pkg.import_module('agent', config=agent_config)
    data_format = am.get_data_format(
        env=env, 
        is_per=replay_config['replay_type'].endswith('per'), 
        n_steps=replay_config['n_steps'])
    process = functools.partial(process_with_env, env=env, one_hot_action=replay_config.get('one_hot_action', True))
    dataset = Dataset(replay, data_format, process_fn=process)
    
    create_model, Agent = pkg.import_agent(config=agent_config)
    models = create_model(model_config, env)
    agent = Agent(
        name=env.name,
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
