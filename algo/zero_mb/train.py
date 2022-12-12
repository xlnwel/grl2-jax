import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.utils import modify_config, TempStore
from tools.timer import Every, Timer
from tools import graph
from tools import pkg
from env.func import create_env
from env.typing import EnvOutput
from algo.zero_mb.run import *


def train(
    configs, 
    agents, 
    model, 
    env, 
    eval_env, 
    buffers, 
    model_buffer
):
    config = configs[0]
    routine_config = config.routine.copy()
    collect_fn = pkg.import_module(
        'elements.utils', algo=routine_config.algorithm).collect
    collects = [functools.partial(collect_fn, buffer) for buffer in buffers]
    collect_fn = pkg.import_module(
        'elements.utils', algo=config.dynamics_name).collect
    model_collect = functools.partial(collect_fn, model_buffer)

    step = agents[0].get_env_step()
    # print("Initial running stats:", 
    #     *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=step, 
        init_next=step != 0, 
        final=routine_config.MAX_STEPS)
    rt = Timer('run')
    tt = Timer('train')

    env_output = env.output()
    env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

    for agent in agents:
        agent.store(**{'time/log_total': 0, 'time/log': 0})

    do_logging('Training starts...')
    train_step = agents[0].get_train_step()
    steps_per_iter = env.n_envs * routine_config.n_steps
    while step < routine_config.MAX_STEPS:
        # do_logging(f'start a new iteration with step: {step} vs {routine_config.MAX_STEPS}')
        start_env_step = agents[0].get_env_step()
        assert buffers[0].size() == 0, buffers[0].size()
        assert buffers[1].size() == 0, buffers[1].size()
        with rt:
            env_outputs = run(
                env, routine_config.n_steps, 
                agents, collects, model_collect, 
                env_outputs, [1], [0])
            env_outputs = run(
                env, routine_config.n_steps, 
                agents, collects, model_collect, 
                env_outputs, [0], [1])

        for i, buffer in enumerate(buffers):
            data = buffer.get_data({
                'state_reset': env_outputs[i].reset
            })
            buffer.move_to_queue(data)
        model_buffer.move_to_queue()

        assert buffers[0].ready(), (buffers[0].size(), len(buffers[0]._queue))
        assert buffers[1].ready(), (buffers[1].size(), len(buffers[1]._queue))
        step += steps_per_iter
        
        time2record = to_record(step)
        if time2record:
            before_info = run_comparisons(eval_env, agents)

        # train agents
        for agent in agents:
            start_train_step = agent.get_train_step()
            with tt:
                agent.train_record()
            
            train_step = agent.get_train_step()
            assert train_step != start_train_step, (start_train_step, train_step)
            agent.set_env_step(step)
            agent.trainer.sync_imaginary_params()

        if time2record:
            after_info = run_comparisons(eval_env, agents)

        # train the model
        model.train_record()

        def get_states():
            state = [a.get_states() for a in agents]
            return state
        
        def set_states(states):
            for a, s in zip(agents, states):
                a.set_states(s)

        # train imaginary agents
        with TempStore(get_states, set_states):
            for _ in range(routine_config.n_imaginary_runs):
                with Timer('imaginary_run'):
                    img_eos = run_on_model(
                        model, agents, collects, routine_config)
                for i, buffer in enumerate(buffers):
                    data = buffer.get_data({
                        'state_reset': img_eos[i].reset
                    })
                    buffer.move_to_queue(data)
                for agent in agents:
                    with Timer('imaginary_train'):
                        agent.imaginary_train()

        if time2record:
            info = {k: np.mean(after_info[k]) - np.mean(before_info[k]) for k in before_info.keys()}
            with Timer('log'):
                for agent in agents:
                    agent.store(**{
                        'stats/train_step': train_step,
                        'time/fps': (step-start_env_step)/rt.last(), 
                        'time/tps': (train_step-start_train_step)/tt.last(),
                    }, **info, **Timer.all_stats())
                    agent.record(step=step)
                    agent.save()
                model.record(step=step)
                model.save()
        # do_logging(f'finish the iteration with step: {step}')


def main(configs, train=train):
    config = configs[0]
    seed = config.get('seed')
    do_logging(f'seed={seed}', level='print')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1 or config.routine.get('EVAL_PERIOD', False)
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    def build_envs():
        env = create_env(config.env, force_envvec=True)
        config.env.seed += 1000
        eval_env = create_env(config.env, force_envvec=True)
        
        return env, eval_env
    
    env, eval_env = build_envs()

    configs, model_config = configs[:-1], configs[-1]
    # load agents
    env_stats = env.stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    agents = []
    buffers = []
    root_dir = config.root_dir
    model_name = config.model_name
    for i, c in enumerate(configs):
        assert c.aid == i, (c.aid, i)
        if f'a{i}' in model_name:
            new_model_name = model_name
        else:
            new_model_name = '/'.join([model_name, f'a{i}'])
        modify_config(
            configs[i], 
            model_name=new_model_name, 
        )
        builder = ElementsBuilder(
            configs[i], 
            env_stats, 
            to_save_code=False, 
            max_steps=config.routine.MAX_STEPS
        )
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        buffers.append(elements.buffer)
    save_code(ModelPath(root_dir, model_name))

    # load model
    new_model_name = '/'.join([model_name, 'model'])
    model_config = modify_config(
        model_config, 
        max_layer=1, 
        aid=0,
        algorithm='magw', 
        root_dir=root_dir, 
        model_name=new_model_name, 
        seed=seed+1000
    )
    builder = ElementsBuilder(
        model_config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch(config=model_config)
    model = elements.agent
    model_buffer = elements.buffer

    train(
        configs, agents, model, env, 
        eval_env, buffers, model_buffer
    )

    do_logging('Training completed')
