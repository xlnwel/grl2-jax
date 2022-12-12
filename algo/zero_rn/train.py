import functools
import numpy as np
import jax
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.mixin.actor import rms2dict
from core.utils import configure_gpu, set_seed, save_code
from core.typing import dict2AttrDict, ModelPath
from tools.utils import batch_dicts, modify_config
from tools.timer import Every, Timer
from tools import pkg
from jax_tools import jax_utils
from env.func import create_env
from env.typing import EnvOutput


def run(
    env, 
    n_steps, 
    agents, 
    collects, 
    env_outputs
):
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

        done_env_ids = [i for i, r in enumerate(neo.reset) if r]

        if done_env_ids:
            info = env.info(done_env_ids)
            if info:
                info = batch_dicts(info, list)
                for agent in agents:
                    agent.store(**info)
        env_outputs = new_env_outputs

    return env_outputs


def run_with_future_opponents(
    env, 
    n_steps, 
    agents, 
    collects, 
    env_outputs
):
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

            done_env_ids = [i for i, r in enumerate(new_env_outputs[aid].reset) if r]

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


def train(
    configs, 
    agents, 
    env, 
    eval_env_config, 
    buffers
):
    for config in configs:
        config.env = eval_env_config
    routine_config = config.routine.copy()
    collect_fn = pkg.import_module(
        'elements.utils', algo=routine_config.algorithm).collect
    collects = [functools.partial(collect_fn, buffer) for buffer in buffers]

    step = agents[0].get_env_step()
    # print("Initial running stats:", 
    #     *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=step, 
        init_next=step != 0, 
        final=routine_config.MAX_STEPS)
    to_eval = Every(
        routine_config.EVAL_PERIOD, 
        start=step, 
        init_next=step != 0, 
        final=routine_config.MAX_STEPS)
    rt = Timer('run')
    tt = Timer('train')
    irt = Timer('imaginary_run')
    itt = Timer('imaginary_train')
    lt = Timer('log')

    eval_process = None
    def evaluate_agent(step):
        eval_main = pkg.import_main('eval', config=config)
        eval_main = ray.remote(eval_main)
        p = eval_main.remote(
            configs, 
            routine_config.N_EVAL_EPISODES, 
            record=routine_config.RECORD_VIDEO, 
            fps=1, 
            info=step // routine_config.EVAL_PERIOD * routine_config.EVAL_PERIOD
        )
        return p

    def record_stats(
        agents, step, start_env_step, train_step, start_train_step):
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
        # do_logging(f'start a new iteration with step: {step} vs {routine_config.MAX_STEPS}')
        start_env_step = agents[0].get_env_step()
        assert buffers[0].size() == 0, buffers[0].size()
        assert buffers[1].size() == 0, buffers[1].size()
        with rt:
            if routine_config.run_with_future_opponents:
                env_outputs = run_with_future_opponents(
                    env, routine_config.n_steps, 
                    agents, collects, env_outputs)
            else:
                env_outputs = run(
                    env, routine_config.n_steps, 
                    agents, collects, env_outputs)
        for buffer in buffers:
            buffer.move_to_queue()

        assert buffers[0].ready(), (buffers[0].size(), len(buffers[0]._queue))
        assert buffers[1].ready(), (buffers[1].size(), len(buffers[1]._queue))
        step += steps_per_iter

        # train agents
        for agent in agents:
            start_train_step = agent.get_train_step()
            with tt:
                agent.train_record()
            
            train_step = agent.get_train_step()
            assert train_step != start_train_step, (start_train_step, train_step)
            agent.set_env_step(step)
            agent.model.sync_imaginary_params()
        
        # perturb imaginary agents
        if routine_config.run_with_future_opponents:
            for agent in agents:
                with itt:
                    agent.model.perturb_imaginary_params()
        
        # evaluation
        if eval_process is not None:
            _, _, video = ray.get(eval_process)
            if config.monitor.use_tensorboard:
                agent.video_summary(video, step=step, fps=1)
        if to_eval(step):
            eval_process = evaluate_agent(step)

        if to_record(step):
            record_stats(
                agents, step, start_env_step, train_step, start_train_step)
        # do_logging(f'finish the iteration with step: {step}')
    if eval_process is None and to_eval(step):
        eval_process = evaluate_agent(step)
    if eval_process is not None:
        ray.get(eval_process)


def main(configs, train=train):
    config = configs[0]
    seed = config.get('seed')
    do_logging(f'seed={seed}', level='print')
    set_seed(seed)

    configure_gpu(None)
    use_ray = bool(config.env.get('n_runners', 1))
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    def build_envs():
        env = create_env(config.env, force_envvec=True)
        eval_env_config = dict2AttrDict(config.env, to_copy=True)
        if config.routine.get('EVAL_PERIOD', False):
            eval_env_config.n_envs = 1
            eval_env_config.n_runners = 1
        
        return env, eval_env_config
    
    env, eval_env_config = build_envs()

    # load agents
    env_stats = env.stats()
    agents = []
    buffers = []
    root_dir = config.root_dir
    model_name = config.model_name
    for i in range(2):
        assert configs[i].aid == i, (configs[i].aid, i)
        if f'a{i}' in model_name:
            new_model_name = model_name
        else:
            new_model_name = '/'.join([model_name, f'a{i}'])
        modify_config(
            configs[i], 
            model_name=new_model_name, 
        )
        builder = ElementsBuilder(configs[i], env_stats, to_save_code=False)
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        buffers.append(elements.buffer)
    save_code(ModelPath(root_dir, model_name))

    train(configs, agents, env, eval_env_config, buffers)

    do_logging('Training completed')
