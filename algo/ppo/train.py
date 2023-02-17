from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code
from core.typing import ModelPath
from tools.display import print_dict, print_dict_info
from tools.store import StateStore
from tools.utils import modify_config
from tools.timer import Every, Timer
from .run import *


def state_constructor_with_sliced_envs(agents, runner):
    agent_states = [a.build_memory() for a in agents]
    env_config = runner.env_config()
    env_config.n_envs //= len(agents)
    runner_states = runner.build_env(env_config)
    return agent_states, runner_states


def state_constructor(agents, runner):
    agent_states = [a.build_memory() for a in agents]
    runner_states = runner.build_env()
    return agent_states, runner_states


def get_states(agents, runner):
    agent_states = [a.get_memory() for a in agents]
    runner_states = runner.get_states()
    return agent_states, runner_states


def set_states(states, agents, runner):
    agent_states, runner_states = states
    assert len(agents) == len(agent_states)
    for a, s in zip(agents, agent_states):
        a.set_memory(s)
    runner.set_states(runner_states)


def lookahead_run(agents, runner, buffers, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

    with Timer('lookahead_run'):
        if routine_config.lookahead_rollout == 'sim':
            with StateStore('sim', constructor, get_fn, set_fn):
                runner.run(
                    routine_config.n_steps, 
                    agents, buffers, 
                    all_aids, all_aids, False, 
                    compute_return=routine_config.compute_return_at_once
                )
        elif routine_config.lookahead_rollout == 'uni':
            for i in all_aids:
                with StateStore(f'uni{i}', constructor, get_fn, set_fn):
                    runner.run(
                        routine_config.n_steps, 
                        agents, buffers, 
                        [i], [i], False, 
                        compute_return=routine_config.compute_return_at_once
                    )
        else:
            raise NotImplementedError

    for i, buffer in enumerate(buffers):
        assert buffer.ready(), f"buffer {i}: ({buffer.size()}, {len(buffer._queue)})"


def lookahead_optimize(agents, routine_config, aids=None):
    if aids is None:
        all_aids = list(range(len(agents)))
        aids = np.random.choice(
            all_aids, size=len(all_aids), replace=False, 
            p=routine_config.perm)
    teammate_log_ratio = None

    for aid in aids:
        agent = agents[aid]
        tlr = agent.lookahead_train(
            teammate_log_ratio=teammate_log_ratio)
        if not routine_config.ignore_ratio_for_lookahead:
            teammate_log_ratio = tlr


def lookahead_train(agents, runner, buffers, routine_config, 
        aids, n_runs, run_fn, opt_fn):
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        run_fn(agents, runner, buffers, routine_config)
        opt_fn(agents, routine_config, aids)


def ego_run(agents, runner, buffers, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

    for i, buffer in enumerate(buffers):
        assert buffer.size() == 0, f"buffer {i}: {buffer.size()}"

    with Timer('run'):
        if routine_config.n_lookahead_steps:
            for i in all_aids:
                lka_aids = [aid for aid in all_aids if aid != i]
                with StateStore(f'real{i}', constructor, get_fn, set_fn):
                    runner.run(
                        routine_config.n_steps, 
                        agents, buffers, 
                        lka_aids, [i], 
                        compute_return=routine_config.compute_return_at_once
                    )
        else:
            with StateStore('real', constructor, get_fn, set_fn):
                runner.run(
                    routine_config.n_steps, 
                    agents, buffers, 
                    [], all_aids, 
                    compute_return=routine_config.compute_return_at_once
                )

    for i, buffer in enumerate(buffers):
        assert buffer.ready(), f"buffer {i}: ({buffer.size()}, {len(buffer._queue)})"

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    for agent in agents:
        agent.add_env_step(env_steps_per_run)

    return agents[0].get_env_step()


def ego_optimize(agents, routine_config, aids=None):
    if aids is None:
        all_aids = list(range(len(agents)))
        aids = np.random.choice(
            all_aids, size=len(all_aids), replace=False, 
            p=routine_config.perm)
        assert set(aids) == set(all_aids), (aids, all_aids)
    teammate_log_ratio = None

    for aid in aids:
        agent = agents[aid]
        tmp_stats = agent.train_record(teammate_log_ratio=teammate_log_ratio)
        if not routine_config.ignore_ratio_for_ego:
            teammate_log_ratio = tmp_stats["teammate_log_ratio"]

        train_step = agent.get_train_step()
        agent.trainer.sync_lookahead_params()
    return train_step


def ego_train(agents, runner, buffers, routine_config, 
        aids, run_fn, opt_fn):
    env_step = run_fn(
        agents, runner, buffers, routine_config)
    train_step = opt_fn(agents, routine_config, aids)

    return env_step, train_step


def eval_and_log(agents, runner, env_step, train_step, routine_config):
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)
    def constructor():
        env_config = runner.env_config()
        if routine_config.n_eval_envs:
            env_config.n_envs = routine_config.n_eval_envs
        agent_states = [a.build_memory() for a in agents]
        runner_states = runner.build_env()
        return agent_states, runner_states

    with Timer('eval'):
        with StateStore('eval', constructor, get_fn, set_fn):
            scores, epslens, _, video = runner.eval_with_video(
                agents, record_video=routine_config.RECORD_VIDEO
            )
    agents[0].store(**{
        'eval_score': np.mean(scores), 
        'eval_epslen': np.mean(epslens), 
    })

    with Timer('save'):
        for agent in agents:
            agent.save()

    with Timer('log'):
        if video is not None:
            agents[0].video_summary(video, step=env_step, fps=1)
        fps = agents[0].get_env_step_intervals() / Timer('run').last()
        tps = agents[0].get_train_step_intervals() / Timer('train').last()
        agents[0].store(**{
                'stats/train_step': train_step, 
                'time/fps': fps, 
                'time/tps': tps, 
            }, 
            **Timer.all_stats()
        )
        agents[0].record(step=env_step)
        for i in range(1, len(agents)):
            agents[i].clear()


def training_aids(all_aids, routine_config):
    return None


def train(
    agents, 
    runner, 
    buffers, 
    routine_config, 
    aids_fn=training_aids,
    lka_run_fn=lookahead_run, 
    lka_opt_fn=lookahead_optimize, 
    lka_train_fn=lookahead_train, 
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
):
    do_logging('Training starts...')
    env_step = agents[0].get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    all_aids = list(range(len(agents)))

    while env_step < routine_config.MAX_STEPS:
        aids = aids_fn(all_aids, routine_config)
        
        lka_train_fn(
            agents, 
            runner, 
            buffers, 
            routine_config, 
            aids=aids, 
            n_runs=routine_config.n_lookahead_steps, 
            run_fn=lka_run_fn, 
            opt_fn=lka_opt_fn
        )
        env_step, train_step = ego_train_fn(
            agents, 
            runner, 
            buffers, 
            routine_config, 
            aids=aids, 
            run_fn=ego_run_fn, 
            opt_fn=ego_opt_fn
        )

        time2record = agents[0].contains_stats('score') \
            and to_record(env_step)
        if time2record:
            eval_and_log(
                agents, runner, env_step, train_step, routine_config)


def main(configs, train=train):
    config = configs[0]
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    runner = Runner(config.env)

    # load agents
    env_stats = runner.env_stats()
    assert len(configs) == env_stats.n_agents, (len(configs), env_stats.n_agents)
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    agents = []
    buffers = []
    root_dir = config.root_dir
    model_name = config.model_name
    for i, c in enumerate(configs):
        assert c.aid == i, (c.aid, i)
        if model_name.endswith(f'a{i}'):
            new_model_name = model_name
        else:
            new_model_name = '/'.join([model_name, f'a{i}'])
        modify_config(
            configs[i], 
            model_name=new_model_name, 
        )
        if c.routine.compute_return_at_once:
            c.buffer.sample_keys += ['advantage', 'v_target']
        builder = ElementsBuilder(
            configs[i], 
            env_stats, 
            to_save_code=False, 
            max_steps=config.routine.MAX_STEPS
        )
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        buffers.append(elements.buffer)
    if seed == 0:
        save_code(ModelPath(root_dir, model_name))

    routine_config = configs[0].routine.copy()
    train(
        agents, 
        runner, 
        buffers, 
        routine_config
    )

    do_logging('Training completed')
