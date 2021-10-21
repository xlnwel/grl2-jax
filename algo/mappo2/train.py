import functools
import numpy as np

from utility.utils import TempStore
from utility.run import Runner, evaluate
from utility.timer import Timer, Every
from utility import pkg
from algo.ppo.train import main


def train(agent, env, eval_env, buffer):
    eu = pkg.import_module('elements.utils', algo=agent.name)
    collect = functools.partial(eu.collect, buffer)
    random_actor = functools.partial(eu.random_actor, env=env)

    em = pkg.import_module(env.name.split("_")[0], pkg='env')
    info_func = em.info_func if hasattr(em, 'info_func') else None

    step = agent.get_env_step()
    runner = Runner(env, agent, step=step, nsteps=agent.N_STEPS, info_func=info_func)
    
    def initialize_rms(step):
        if step == 0 and agent.actor.is_obs_normalized:
            print('Start to initialize running stats...')
            for i in range(10):
                runner.run(action_selector=random_actor, step_fn=collect)
                life_mask = np.concatenate(buffer['life_mask']) \
                    if env.use_life_mask else None
                agent.actor.update_obs_rms(np.concatenate(buffer['obs']), mask=life_mask)
                agent.actor.update_obs_rms(np.concatenate(buffer['global_state']), 
                    'global_state', mask=life_mask)
                discount = np.logical_and(buffer['discount'], 1 - buffer['reset'])
                agent.actor.update_reward_rms(buffer['reward'], discount)
                buffer.reset()
            buffer.clear()
            agent.set_env_step(runner.step)
            agent.save(print_terminal_info=True)
        return step

    step = initialize_rms(step)

    runner.step = step
    # print("Initial running stats:", *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(agent.LOG_PERIOD, agent.LOG_PERIOD)
    to_eval = Every(agent.EVAL_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    et = Timer('eval')
    lt = Timer('log')

    def evaluate_agent(step, eval_env, agent):
        if eval_env is not None:
            with TempStore(agent.model.get_states, agent.model.reset_states):
                with et:
                    eval_score, eval_epslen, video = evaluate(
                        eval_env, agent, n=agent.N_EVAL_EPISODES, 
                        record_video=agent.RECORD_VIDEO, size=(64, 64))
                if agent.RECORD_VIDEO:
                    agent.video_summary(video, step=step)
                agent.store(
                    eval_score=eval_score, 
                    eval_epslen=eval_epslen)

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
    while step < agent.MAX_STEPS:
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
        discount = np.logical_and(buffer['discount'], 1 - buffer['reset'])
        agent.actor.update_reward_rms(buffer['reward'], discount)
        buffer.update('reward', agent.actor.normalize_reward(buffer['reward']), field='all')
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

        if to_eval(agent.get_train_step()) or step > agent.MAX_STEPS:
            evaluate_agent(step, eval_env, agent)

        if to_record(agent.get_train_step()) and agent.contains_stats('score'):
            record_stats(step)

main = functools.partial(main, train=train)
