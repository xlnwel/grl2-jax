import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.mixin.actor import rms2dict
from core.utils import configure_gpu, set_seed
from core.typing import dict2AttrDict
from tools.utils import TempStore, batch_dicts, modify_config
from tools.run import Runner
from tools.timer import Every, Timer
from tools import pkg
from jax_tools import jax_utils
from env.func import create_env
from env.typing import EnvOutput


def run(env, n_steps, agents, collects, env_outputs):
    # print('raw run')
    for a in agents:
        assert a.strategy.model.params.imaginary == False, a.strategy.model.params.imaginary
        assert a.strategy.model.imaginary_params.imaginary == True, a.strategy.model.imaginary_params.imaginary

    for _ in range(n_steps):
        for a in agents:
            assert a.strategy.model.params.imaginary == False, a.strategy.model.params.imaginary
            
        acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

        action = np.concatenate(acts, axis=-1)
        env_output = env.step(action)
        new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

        for eo, act, stat, neo, collect in zip(
                env_outputs, acts, stats, new_env_outputs, collects):
            kwargs = dict(
                obs=eo.obs, 
                action=act, 
                reward=neo.reward, 
                discount=neo.discount, 
                next_obs=neo.obs, 
                **stat
            )
            collect(env, 0, neo.reset, **kwargs)

        done_env_ids = [i for i, r in enumerate(neo.reset)if r]

        if done_env_ids:
            info = env.info(done_env_ids)
            if info:
                info = batch_dicts(info, list)
                for agent in agents:
                    agent.store(**info)
        env_outputs = new_env_outputs

    return env_outputs


def run_with_future_opponents(env, n_steps, agents, collects, env_outputs):
    # print('run with future agents')
    for aid, agent in enumerate(agents):
        for i, a2 in enumerate(agents):
            if i != aid:
                a2.strategy.model.switch_params()
                assert a2.strategy.model.params.imaginary == True, a2.strategy.model.params.imaginary
                assert a2.strategy.model.imaginary_params.imaginary == False, a2.strategy.model.imaginary_params.imaginary
        for _ in range(n_steps):
            for i, a in enumerate(agents):
                if i != aid:
                    assert a.strategy.model.params.imaginary == True, a.strategy.model.params.imaginary
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = np.concatenate(acts, axis=-1)
            env_output = env.step(action)
            new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

            kwargs = dict(
                obs=env_outputs[aid].obs, 
                action=acts[aid], 
                reward=new_env_outputs[aid].reward, 
                discount=new_env_outputs[aid].discount, 
                next_obs=new_env_outputs[aid].obs, 
                **stats[aid]
            )
            collects[aid](env, 0, new_env_outputs[aid].reset, **kwargs)

            done_env_ids = [i for i, r in enumerate(new_env_outputs[aid].reset)if r]

            if done_env_ids:
                info = env.info(done_env_ids)
                if info:
                    info = batch_dicts(info, list)
                    agent.store(**info)
            env_outputs = new_env_outputs
        for i, a2 in enumerate(agents):
            if i != aid:
                a2.strategy.model.switch_params()
                assert a2.strategy.model.params.imaginary == False, a2.strategy.model.params.imaginary
                assert a2.strategy.model.imaginary_params.imaginary == True, a2.strategy.model.imaginary_params.imaginary
        
    return env_outputs

def run_imaginary_agent(env, n_steps, agents, collects, env_outputs):
    # print('run with future agents')
    for aid, agent in enumerate(agents):
        agent.strategy.model.switch_params()
        assert agent.strategy.model.params.imaginary == True, agent.strategy.model.params.imaginary
        assert agent.strategy.model.imaginary_params.imaginary == False, agent.strategy.model.imaginary_params.imaginary
        for _ in range(n_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = np.concatenate(acts, axis=-1)
            env_output = env.step(action)
            new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

            kwargs = dict(
                obs=env_outputs[aid].obs, 
                action=acts[aid], 
                reward=new_env_outputs[aid].reward, 
                discount=new_env_outputs[aid].discount, 
                next_obs=new_env_outputs[aid].obs, 
                **stats[aid]
            )
            collects[aid](env, 0, new_env_outputs[aid].reset, **kwargs)

            done_env_ids = [i for i, r in enumerate(new_env_outputs[aid].reset)if r]

            if done_env_ids:
                info = env.info(done_env_ids)
                if info:
                    info = batch_dicts(info, list)
                    agent.store(**info)
            env_outputs = new_env_outputs
        agent.strategy.model.switch_params()
        assert agent.strategy.model.params.imaginary == False, agent.strategy.model.params.imaginary
        assert agent.strategy.model.imaginary_params.imaginary == True, agent.strategy.model.imaginary_params.imaginary
        
    return env_outputs


def train(config, agents, env, eval_env_config, buffers):
    routine_config = config.routine
    config.env = eval_env_config
    collect_fn = pkg.import_module(
        'elements.utils', algo=routine_config.algorithm).collect
    collects = [functools.partial(collect_fn, buffer) for buffer in buffers]

    step = agents[0].get_env_step()
    # print("Initial running stats:", 
    #     *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=1000, 
        final=routine_config.MAX_STEPS)
    rt = Timer('run')
    tt = Timer('train')
    lt = Timer('log')

    def record_stats(step, start_env_step, train_step, start_train_step):
        for agent in agents:
            with lt:
                agent.store(**{
                    'stats/train_step': agent.get_train_step(),
                    'time/fps': (step-start_env_step)/rt.last(), 
                    'time/tps': (train_step-start_train_step)/tt.last(),
                }, **Timer.all_stats())
                agent.record(step=step)
                agent.save()

    do_logging('Training starts...')
    train_step = agents[0].get_train_step()
    env_output = env.output()
    env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
    steps_per_iter = env.n_envs * routine_config.n_steps
    while step < routine_config.MAX_STEPS:
        start_env_step = agents[0].get_env_step()
        assert buffers[0].size() == 0, buffers[0].size()
        assert buffers[1].size() == 0, buffers[1].size()
        with rt:
            if routine_config.run_with_future_opponents:
                env_outputs = run_with_future_opponents(
                    env, routine_config.n_steps, agents, collects, env_outputs)
            else:
                env_outputs = run(env, routine_config.n_steps, agents, collects, env_outputs)

        step += steps_per_iter
        for agent in agents:
            start_train_step = agent.get_train_step()
            with tt:
                agent.train_record()
            
            train_step = agent.get_train_step()
            agent.set_env_step(step)
            agent.trainer.sync_imaginary_params()
        if routine_config.run_with_future_opponents:
            for _ in range(routine_config.n_imaginary_runs):
                env_outputs = run_imaginary_agent(
                    env, routine_config.n_steps, agents, collects, env_outputs)

                for agent in agents:
                    agent.imaginary_train()
        
        if to_record(step):
            record_stats(step, start_env_step, train_step, start_train_step)


def main(configs, train=train, gpu=-1):
    assert len(configs) == 1, configs
    config = configs[0]
    seed = config.get('seed')
    do_logging(f'seed={seed}', level='print')
    set_seed(seed)
    configure_gpu()
    use_ray = config.routine.get('EVAL_PERIOD', False)
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    def build_envs():
        env = create_env(config.env, force_envvec=True)
        eval_env_config = config.env.copy()
        if config.routine.get('EVAL_PERIOD', False):
            if config.env.env_name.startswith('procgen'):
                if 'num_levels' in eval_env_config:
                    eval_env_config['num_levels'] = 0
                if 'seed' in eval_env_config \
                    and eval_env_config['seed'] is not None:
                    eval_env_config['seed'] += 1000
                for k in list(eval_env_config.keys()):
                    # pop reward hacks
                    if 'reward' in k:
                        eval_env_config.pop(k)
            else:
                eval_env_config['n_envs'] = 1
            eval_env_config['n_runners'] = 1
        
        return env, eval_env_config
    
    env, eval_env_config = build_envs()

    env_stats = env.stats()
    agents = []
    buffers = []
    for aid in range(2):
        c = dict2AttrDict(config, to_copy=True)
        model_name = '/'.join([config.model_name, f'a{aid}'])
        c = modify_config(
            c, model_name=model_name, 
        )
        builder = ElementsBuilder(c, env_stats, to_save_code=True)
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        buffers.append(elements.buffer)

    train(config, agents, env, eval_env_config, buffers)
