import functools
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.store import StateStore
from tools.utils import modify_config
from tools.timer import Every, Timer
from tools import pkg
from algo.lka_v2.run import *


def train(
    configs,
    agents,
    runner,
    buffers
):
    def state_constructor():
        agent_states = [a.build_memory() for a in agents]
        runner_states = runner.build_env()
        return agent_states, runner_states
    
    def get_state():
        agent_states = [a.get_memory() for a in agents]
        runner_states = runner.get_states()
        return agent_states, runner_states
    
    def set_states(states):
        agent_states, runner_states = states
        assert len(agents) == len(agent_states)
        for a, s in zip(agents, agent_states):
            a.set_memory(s)
        runner.set_states(runner_states)
        
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
    to_eval = Every(
        routine_config.EVAL_PERIOD, 
        start=step, 
        final=routine_config.MAX_STEPS)
    rt = Timer('run')
    tt = Timer('train')

    def evaluate_agent(step):
        if to_eval(step):
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
        else:
            return None
    eval_process = evaluate_agent(step)

    for agent in agents:
        agent.store(**{'time/log_total': 0, 'time/log': 0})

    do_logging('Training starts...')
    train_step = agents[0].get_train_step()
    env_stats = runner.env_stats()
    steps_per_iter = env_stats.n_envs * routine_config.n_steps
    # eval_info = {}
    # diff_info = {}
    # with StateStore('comp', state_constructor, get_state, set_states):
    #     prev_info = run_comparisons(runner, agents)
    while step < routine_config.MAX_STEPS:
        # train imaginary agents
        for _ in range(routine_config.n_imaginary_runs):
            with Timer('imaginary_run'):
                env_outputs = [None for _ in range(len(agents))]
                for idx in range(len(agents)):
                    with StateStore(f'img{idx}',
                        state_constructor,
                        get_state, set_states
                    ):
                        env_outputs[idx] = runner.run(
                            routine_config.n_steps,
                            agents, collects,
                            [i for i in range(len(agents)) if i != idx], [idx]
                        )[idx]
            for i, buffer in enumerate(buffers):
                data = buffer.get_data({
                    'state_reset': env_outputs[i].reset
                })
                buffer.move_to_queue(data)
            for agent in agents:
                with Timer('imaginary_train'):
                    agent.imaginary_train()

        # do_logging(f'start a new iteration with step: {step} vs {routine_config.MAX_STEPS}')
        start_env_step = agents[0].get_env_step()
        for i, buffer in enumerate(buffers):
            assert buffer.size() == 0, f"buffer i: {buffer.size()}"
        with rt:
            env_outputs = [None for _ in range(len(agents))]
            for idx in range(len(agents)):
                with StateStore(f'real{idx}',
                    state_constructor,
                    get_state, set_states
                ):
                    env_outputs[idx] = runner.run(
                        routine_config.n_steps,
                        agents, collects,
                        [i for i in range(len(agents)) if i != idx], [idx]
                    )[idx]
            for i, buffer in enumerate(buffers):
                data = buffer.get_data({
                    'state_reset': env_outputs[i].reset
                })
                buffer.move_to_queue(data)
        
        for buffer in buffers:
            assert buffer.ready(), f"buffer i: ({buffer.size()}, {len(buffer._queue)})"
            
        step += steps_per_iter

        time2record = to_record(step)
        # if time2record:
        #     with StateStore('comp', state_constructor, get_state, set_states):
        #         before_info = run_comparisons(runner, agents)

        # train agents
        for agent in agents:
            start_train_step = agent.get_train_step()
            with tt:
                agent.train_record()
            
            train_step = agent.get_train_step()
            assert train_step != start_train_step, (start_train_step, train_step)
            agent.set_env_step(step)
            agent.trainer.sync_imaginary_params()

        # if time2record:
        #     with StateStore('comp', state_constructor, get_state, set_states):
        #         after_info = run_comparisons(runner, agents)
        
        if time2record:
            # info = {
            #     f'diff_{k}': after_info[k] - before_info[k] 
            #     for k in before_info.keys()
            # }
            # info.update({
            #     f'dist_diff_{k}': after_info[k] - prev_info[k] 
            #     for k in before_info.keys()
            # })
            # eval_info = batch_dicts([eval_info, after_info])
            # diff_info = batch_dicts([diff_info, info])
            # prev_info = after_info
            if eval_process is not None:
                scores, epslens, video = ray.get(eval_process)
                for agent in agents:
                    agent.store(**{
                        'metrics/eval_score': np.mean(scores), 
                        'metrics/eval_epslen': np.mean(epslens), 
                    })
                agent.video_summary(video, step=step, fps=1)
            eval_process = evaluate_agent(step)
            with Timer('log'):
                for agent in agents:
                    agent.store(**{
                        'stats/train_step': train_step,
                        'time/fps': (step-start_env_step)/rt.last(), 
                        'time/tps': (train_step-start_train_step)/tt.last(),
                    }, 
                    # **eval_info, **diff_info, 
                    **Timer.all_stats())
                    agent.record(step=step)
                    agent.save()
        # do_logging(f'finish the iteration with step: {step}')


def main(configs, train=train):
    config = configs[0]
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1 or config.routine.get('EVAL_PERIOD', False)
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    runner = Runner(config.env)

    # load agents
    env_stats = runner.env_stats()
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

    train(configs, agents, runner, buffers)

    do_logging('Training completed')
