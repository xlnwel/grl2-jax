import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.utils import modify_config
from tools.timer import Every, Timer
from tools import graph
from tools import pkg
from env.func import create_env
from env.typing import EnvOutput
from algo.zero.run import *


def train(
    configs, 
    agents, 
    env, 
    envs, 
    eval_env, 
    buffers
):
    config = configs[0]
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
    rt = Timer('run')
    tt = Timer('train')

    env_output = env.output()
    env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
    env_output1 = envs[0].output()
    env_outputs1 = [EnvOutput(*o) for o in zip(*env_output1)]
    env_output2 = envs[1].output()
    env_outputs2 = [EnvOutput(*o) for o in zip(*env_output2)]

    agent_tracks = None
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
            env_outputs1, ats = run(
                envs[0], routine_config.n_steps, 
                agents, collects, env_outputs1, 
                [1], [0, 1])
            if ats is not None:
                agent_tracks = ats if agent_tracks is None else \
                    [sum([at1, at2]) for at1, at2 in zip(agent_tracks, ats)]
                ats = None
            for buffer in buffers:
                buffer.move_to_queue()
            env_outputs2, ats = run(
                envs[1], routine_config.n_steps, 
                agents, collects, env_outputs2, 
                [0], [0, 1])
        if ats is not None:
            agent_tracks = ats if agent_tracks is None else \
                [sum([at1, at2]) for at1, at2 in zip(agent_tracks, ats)]

        for buffer in buffers:
            buffer.move_to_queue()

        assert buffers[0].ready(), (buffers[0].size(), len(buffers[0]._queue))
        assert buffers[1].ready(), (buffers[1].size(), len(buffers[1]._queue))
        step += steps_per_iter

        time2record = agent_tracks is not None and to_record(step)
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

        # train imaginary agents
        for _ in range(routine_config.n_imaginary_runs):
            with Timer('imaginary_run'):
                env_outputs, _ = run(
                    env, routine_config.n_steps, 
                    agents, collects, env_outputs, 
                    [0, 1], [0, 1], False)
            for buffer in buffers:
                buffer.move_to_queue()
            for agent in agents:
                with Timer('imaginary_train'):
                    agent.imaginary_train()
        
        if time2record:
            info = {k: np.mean(after_info[k]) - np.mean(before_info[k]) for k in before_info.keys()}
            xticklabels = graph.get_tick_labels(agent_tracks[0].shape[0])
            yticklabels = graph.get_tick_labels(agent_tracks[0].shape[1])
            with Timer('log'):
                for agent, track in zip(agents, agent_tracks):
                    track_prob = track / np.sum(track)
                    agent.matrix_summary(
                        matrix=track_prob, 
                        xlabel='x', 
                        ylabel='y', 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels, 
                        name='track'
                    )
                    track_entropy = np.sum(-track_prob * np.log(track_prob))
                    agent.store(**{
                        'metrics/track_entropy': track_entropy, 
                        'stats/train_step': agent.get_train_step(),
                        'time/fps': (step-start_env_step)/rt.last(), 
                        'time/tps': (train_step-start_train_step)/tt.last(),
                    }, **info, **Timer.all_stats())
                    agent.record(step=step)
                    agent.save()
            agent_tracks = None
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
        env_config = config.env.copy()
        env = create_env(env_config, force_envvec=True)
        env_config.n_envs //= 2
        envs = []
        for _ in range(2):
            env_config.seed += 100
            envs.append(create_env(env_config, force_envvec=True))
        config.env.seed += 1000
        eval_env = create_env(config.env, force_envvec=True)
        
        return env, envs, eval_env
    
    env, envs, eval_env = build_envs()

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

    train(configs, agents, env, envs, eval_env, buffers)

    do_logging('Training completed')
