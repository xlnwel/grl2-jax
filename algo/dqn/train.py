import functools
import numpy as np
import tensorflow as tf
import ray

from core.tf_config import *
from utility.utils import Every
from utility.rl_utils import compute_act_temp, compute_act_eps
from utility.graph import video_summary
from utility.ray_setup import sigint_shutdown_ray
from utility.timer import Timer
from utility.run import Runner, evaluate
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from core.dataset import Dataset, process_with_env


def train(agent, env, eval_env, replay):
    def collect(env, env_step, reset, **kwargs):
        # if reset:
        #     # we reset noisy every episode. Theoretically, 
        #     # this follows the guide of deep exploration.
        #     # More importantly, it saves time!
        #     if hasattr(agent, 'reset_noisy'):
        #         agent.reset_noisy()
        replay.add(**kwargs)
    
    env_step = agent.env_step
    runner = Runner(env, agent, step=env_step, nsteps=agent.TRAIN_PERIOD)
    while not replay.good_to_learn():
        env_step = runner.run(step_fn=collect)

    to_eval = Every(agent.EVAL_PERIOD)
    to_log = Every(agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    to_record = Every(agent.EVAL_PERIOD*10)
    rt = Timer('run')
    tt = Timer('train')
    print('Training starts...')
    while env_step <= int(agent.MAX_STEPS):
        with rt:
            env_step = runner.run(step_fn=collect)
        with tt:
            agent.learn_log(env_step)

        if to_eval(env_step):
            record = agent.RECORD and to_record(env_step)
            eval_score, eval_epslen, video = evaluate(
                eval_env, agent, record=record, n=agent.N_EVAL_EPISODES)
            if record:
                video_summary(f'{agent.name}/sim', video, step=env_step)
            agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
        if to_log(env_step):
            fps = rt.average() * agent.TRAIN_PERIOD
            tps = tt.average() * agent.N_UPDATES
            
            agent.store(
                env_step=agent.env_step,
                train_step=agent.train_step,
                fps=fps, 
                tps=tps,
            )
            agent.log(env_step)
            agent.save()

def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(agent_config.get('precision', 32))

    use_ray = env_config.get('n_workers', 1) > 1
    if use_ray:
        ray.init()
        sigint_shutdown_ray()

    n_workers = env_config.get('n_workers', 1)
    n_envs = env_config.get('n_envs', 1)
    if n_envs > 1:
        agent_config = compute_act_eps(
            agent_config, None, n_workers, 
            env_config['n_envs'])
    if 'actor' in model_config:
        model_config = compute_act_temp(
            agent_config, model_config, None, 
            n_workers, n_envs)
    
    env = create_env(env_config)
    # if env_config['name'].startswith('procgen'):
    #     start_level = 200
    eval_env_config = env_config.copy()
    eval_env_config['n_workers'] = 1
    eval_env_config['n_envs'] = 64 if 'procgen' in eval_env_config['name'] else 1
    eval_env_config['np_obs'] = True
    reward_key = [k for k in eval_env_config.keys() if 'reward' in k]
    [eval_env_config.pop(k) for k in reward_key]
    eval_env = create_env(eval_env_config)
    replay_config['n_envs'] = n_workers * n_envs
    replay = create_replay(replay_config)

    am = pkg.import_module('agent', config=agent_config)
    data_format = am.get_data_format(
        env=env, 
        is_per=replay_config['replay_type'].endswith('per'), 
        n_steps=replay_config['n_steps'])
    process = functools.partial(process_with_env, 
        env=env, one_hot_action=replay_config.get('one_hot_action', True))
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

    if use_ray:
        ray.shutdown()

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

#         next_obs, reward, discount, reset = env.env_step(action)
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
#             agent.log(env_step=t)
#             agent.save()

# def evaluate(env, agent):
#     score = 0
#     epslen = 0
#     max_steps = 27000
#     i = 0
#     obs = env.reset().obs
#     discount = 1
#     while discount and i < max_steps:
#         action = agent(obs, evaluation=True)
#         obs, reward, discount, reset = env.env_step(action)
#         score += reward
#         epslen += 1
#         i += 1
#     assert score == env.score(), f'{score} vs {env.score()}'

#     return score, epslen
