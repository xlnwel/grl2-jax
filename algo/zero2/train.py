import functools
import numpy as np

from core.elements.builder import ElementsBuilder
from core.mixin.actor import rms2dict
from core.tf_config import \
    configure_gpu, configure_precision, silence_tf_logs
from utility.display import pwt
from utility.utils import TempStore, set_seed
from utility.run import Runner, evaluate
from utility.timer import Every, Timer
from utility import pkg
from env.func import create_env


def train(config, agent, env, eval_env, buffer):
    collect_fn = pkg.import_module(
        'elements.utils', algo=config.algorithm).collect
    collect = functools.partial(collect_fn, buffer)

    suite_name = env.name.split("-")[0] \
        if '-' in env.name else 'gym'
    em = pkg.import_module(suite_name, pkg='env')
    info_func = em.info_func if hasattr(em, 'info_func') else None

    step = agent.get_env_step()
    runner = Runner(
        env, agent, step=step, nsteps=config.n_steps, info_func=info_func)

    def initialize_rms():
        print('Start to initialize running stats...')
        for _ in range(10):
            runner.run(action_selector=env.random_action, step_fn=collect)
            agent.actor.update_obs_rms({'obs': buffer['obs']})
            agent.actor.update_reward_rms(
                np.array(buffer['reward']), np.array(buffer['discount']))
            buffer.reset()
        buffer.clear()
        agent.set_env_step(runner.step)
        agent.save(print_terminal_info=True)

    if step == 0 and agent.actor.is_obs_normalized:
        initialize_rms()

    runner.step = step
    # print("Initial running stats:", 
    #     *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(config.LOG_PERIOD)
    to_eval = Every(config.EVAL_PERIOD)
    rt = Timer('run')
    tt = Timer('train')
    et = Timer('eval')
    lt = Timer('log')

    def evaluate_agent(step, eval_env, agent):
        if eval_env is not None:
            with TempStore(agent.model.get_states, agent.model.reset_states):
                with et:
                    eval_score, eval_epslen, video = evaluate(
                        eval_env, agent, n=config.N_EVAL_EPISODES, 
                        record_video=config.RECORD_VIDEO, size=(64, 64))
                if config.RECORD_VIDEO:
                    agent.video_summary(video, step=step)
                agent.store(
                    eval_score=eval_score, 
                    eval_epslen=eval_epslen)

    def record_stats(step):
        aux_stats = agent.actor.get_rms_stats()
        aux_stats = rms2dict(aux_stats)
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
            }, **aux_stats)
            agent.record(step=step)
            agent.save()

    pwt('Training starts...')
    train_step = agent.get_train_step()
    while step < config.MAX_STEPS:
        start_env_step = agent.get_env_step()
        assert buffer.size() == 0, buffer.size()
        with rt:
            step = runner.run(step_fn=collect)
        # reward normalization
        reward = np.array(buffer['reward'])
        discount = np.array(buffer['discount'])
        agent.actor.update_reward_rms(reward, discount)
        buffer.update(
            'reward', agent.actor.normalize_reward(reward), field='all')
        agent.record_inputs_to_vf(runner.env_output)
        
        # observation normalization
        raw_obs = buffer['obs']
        obs = agent.actor.normalize_obs(raw_obs)
        buffer.update('obs', obs, field='all')
        next_obs = agent.actor.normalize_obs(buffer['next_obs'])
        agent.actor.update_obs_rms(raw_obs)
        if not config.get('timeout_done', True):
            value = agent.compute_value({'global_state': obs})
            next_value = agent.compute_value({'global_state': next_obs})
            buffer.finish(
                value=value.reshape(buffer.n_steps, buffer.n_envs), 
                next_value=next_value.reshape(buffer.n_steps, buffer.n_envs),
                reset=buffer['reset']
            )
        else:
            value = agent.compute_value()
            buffer.finish(last_value=value)

        start_train_step = agent.get_train_step()
        with tt:
            agent.train_record()
        train_step = agent.get_train_step()
        agent.store(
            fps=(step-start_env_step)/rt.last(),
            tps=(train_step-start_train_step)/tt.last())
        agent.set_env_step(step)
        buffer.reset()
        if to_eval(train_step) or step > config.MAX_STEPS:
            evaluate_agent(step, eval_env, agent)

        if to_record(train_step) and agent.contains_stats('score'):
            record_stats(step)

def main(configs, train=train):
    assert len(configs) == 1, configs
    config = configs[0]

    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        import ray
        from utility.ray_setup import sigint_shutdown_ray
        ray.init()
        sigint_shutdown_ray()

    silence_tf_logs()
    seed = config.get('seed')
    print('seed', seed)
    set_seed(seed)
    configure_gpu()
    configure_precision(config.precision)

    def build_envs():
        env = create_env(config.env, force_envvec=True)
        eval_env_config = config.env.copy()
        if config.env.pop('do_evaluation', False):
            if 'num_levels' in eval_env_config:
                eval_env_config['num_levels'] = 0
            if 'seed' in eval_env_config \
                and eval_env_config['seed'] is not None:
                eval_env_config['seed'] += 1000
            eval_env_config['n_runners'] = 1
            for k in list(eval_env_config.keys()):
                # pop reward hacks
                if 'reward' in k:
                    eval_env_config.pop(k)
            
            eval_env = create_env(eval_env_config, force_envvec=True)
        else: 
            eval_env = None
        
        return env, eval_env
    
    env, eval_env = build_envs()

    env_stats = env.stats()
    builder = ElementsBuilder(config, env_stats)
    elements = builder.build_agent_from_scratch()

    train(config.routine, elements.agent, env, eval_env, elements.buffer)

    if use_ray:
        env.close()
        if eval_env is not None:
            eval_env.close()
        ray.shutdown()
