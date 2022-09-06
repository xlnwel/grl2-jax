import functools
import numpy as np

from tools.run import Runner
from tools.timer import Timer, Every
from tools import pkg
from algo.ppo.train import main


def train(config, agent, env, eval_env, buffer):
    eu = pkg.import_module('elements.utils', algo=config.algorithm)
    collect = functools.partial(eu.collect, buffer)
    random_actor = functools.partial(eu.random_actor, env=env)

    suite_name = env.name.split("-")[0] \
        if '-' in env.name else 'builtin'
    em = pkg.import_module(suite_name, pkg='env')
    info_func = em.info_func if hasattr(em, 'info_func') else None

    step = agent.get_env_step()
    runner = Runner(env, agent, step=step, nsteps=config.n_steps, info_func=info_func)
    
    def initialize_rms(step):
        if step == 0 and agent.actor.is_obs_normalized:
            print('Start to initialize running stats...')
            for i in range(10):
                runner.run(action_selector=random_actor, step_fn=collect)
                agent.actor.update_all_rms(
                    buffer, buffer['life_mask'] if env.use_life_mask else None,
                    axis=(0, 1))
                buffer.reset()
            buffer.clear()
            agent.set_env_step(runner.step)
            agent.save(print_terminal_info=True)
            return step

    initialize_rms(step)

    runner.step = step
    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(config.LOG_PERIOD, config.LOG_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    et = Timer('eval')
    lt = Timer('log')


    def record_stats(step):
        with lt:
            agent.store(**{
                'misc/train_step': agent.get_train_step(),
                'time/run': rt.total(), 
                'time/train': tt.total(),
                'time/eval': et.total(),
                'time/log': lt.total(),
                'time/run_mean': rt.average(), 
                'time/train_mean': tt.average(),
                'time/eval_mean': et.average(),
                'time/log_mean': lt.average(),
            })
            agent.record(step=step)
            agent.save()

    print('Training starts...')
    while step < config.MAX_STEPS:
        start_env_step = agent.get_env_step()
        with rt:
            step = runner.run(step_fn=collect)
        # NOTE: normalizing rewards here may introduce some inconsistency 
        # if normalized rewards is fed as an input to the network.
        # One can reconcile this by moving normalization to collect 
        # or feeding the network with unnormalized rewards.
        # The latter is adopted in our implementation. 
        # However, the following line currently doesn't store
        # a copy of unnormalized rewards
        agent.actor.update_reward_rms(
            buffer['reward'], buffer['discount'])
        buffer.update(
            'reward', agent.actor.normalize_reward(buffer['reward']), field='all')
        agent.record_inputs_to_vf(runner.env_output)
        value = agent.compute_value()
        buffer.finish(value)

        start_train_step = agent.get_train_step()
        with tt:
            agent.train_record()
        agent.store(
            fps=(step-start_env_step)/rt.last(),
            tps=(agent.get_train_step()-start_train_step)/tt.last())
        agent.set_env_step(step)
        buffer.reset()

        if to_record(agent.get_train_step()) and agent.contains_stats('score'):
            record_stats(step)

main = functools.partial(main, train=train)
