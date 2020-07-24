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
            agent.learn_log(step)
    
    step = agent.env_step
    collect = lambda *args, **kwargs: replay.add(**kwargs)
    runner = Runner(env, agent, step=step, nsteps=agent.LOG_PERIOD)
    while not replay.good_to_learn():
        step = runner.run(
            action_selector=env.random_action, 
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
        agent.log(step)
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
    process = functools.partial(process_with_env, env=env)
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

    # This training process is used for Mujoco tasks, following the same process as OpenAI's spinningup
    # print('hey, v2')
    # from tmp import make_atari, wrap_deepmind
    # env = make_atari('MsPacmanNoFrameskip-v4')
    # env = wrap_deepmind(env, life_done=True)
    # eval_env = make_atari('MsPacmanNoFrameskip-v4')
    # eval_env = wrap_deepmind(eval_env, life_done=False, clip_rewards=False)
    # obs = env.reset()
#     obs = env.output().obs
#     score = 0
#     epslen = 0
#     from utility.utils import Every
#     to_log = Every(agent.LOG_PERIOD, start=3e4)
#     for t in range(int(agent.MAX_STEPS)):
#         if t > 20000:
#             action = agent(obs)
#         else:
#             action = env.action_space.sample()

#         next_obs, reward, discount, reset = env.step(action)
#         # discount = np.float32(1-done)
#         epslen += 1
#         score += reward
#         replay.add(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs)
#         obs = next_obs
#         # if done:
#         #     if env.ale.lives() == 0:
#         #         agent.store(score=score, epslen=epslen)
#         #         epslen = 0
#         #         score = 0
#         #     obs = env.reset()
#         if reset:
#             agent.store(score=env.score(), epslen=env.epslen())
#             score = 0
#             epslen = 0

#         if replay.good_to_learn() and t % 4 == 0:
#             agent.learn_log(t)

#         if to_log(t):
#             eval_score, eval_epslen = evaluate(eval_env, agent)

#             agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
#             agent.log(step=t)
#             agent.save()

# def evaluate(env, agent):
#     score = 0
#     epslen = 0
#     max_steps = 27000
#     i = 0
#     obs = env.reset().obs
#     discount = 1
#     while discount and i < max_steps:
#         action = agent(obs, deterministic=True)
#         obs, reward, discount, reset = env.step(action)
#         score += reward
#         epslen += 1
#         i += 1
#     assert score == env.score(), f'{score} vs {env.score()}'

#     return score, epslen
