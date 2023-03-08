from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.utils import configure_gpu, set_seed, save_code_for_seed
from tools.display import print_dict, print_dict_info
from tools.store import StateStore
from tools.utils import modify_config
from tools.timer import Every, Timer, timeit
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


@timeit
def lookahead_run(agents, runner, buffers, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

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


@timeit
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


@timeit
def lookahead_train(agents, runner, buffers, routine_config, 
        aids, n_runs, run_fn, opt_fn):
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        run_fn(agents, runner, buffers, routine_config)
        opt_fn(agents, routine_config, aids)


@timeit
def ego_run(agents, runner, buffers, routine_config):
    all_aids = list(range(len(agents)))
    constructor = partial(state_constructor, agents=agents, runner=runner)
    get_fn = partial(get_states, agents=agents, runner=runner)
    set_fn = partial(set_states, agents=agents, runner=runner)

    for i, buffer in enumerate(buffers):
        assert buffer.size() == 0, f"buffer {i}: {buffer.size()}"

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


@timeit
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


@timeit
def ego_train(agents, runner, buffers, routine_config, 
        aids, run_fn, opt_fn):
    env_step = run_fn(
        agents, runner, buffers, routine_config)
    train_step = opt_fn(agents, routine_config, aids)

    return env_step, train_step


@timeit
def evaluate(agents, runner, env_step, routine_config):
    if routine_config.EVAL_PERIOD:
        get_fn = partial(get_states, agents=agents, runner=runner)
        set_fn = partial(set_states, agents=agents, runner=runner)
        def constructor():
            env_config = runner.env_config()
            if routine_config.n_eval_envs:
                env_config.n_envs = routine_config.n_eval_envs
            agent_states = [a.build_memory() for a in agents]
            runner_states = runner.build_env()
            return agent_states, runner_states

        with StateStore('eval', constructor, get_fn, set_fn):
            eval_scores, eval_epslens, _, video = runner.eval_with_video(
                agents, record_video=routine_config.RECORD_VIDEO
            )

        agents[0].store(**{
            'eval_score': np.mean(eval_scores), 
            'eval_epslen': np.mean(eval_epslens), 
        })
        if video is not None:
            agents[0].video_summary(video, step=env_step, fps=1)


@timeit
def save(agents):
    for agent in agents:
        agent.save()


@timeit
def log(agents, env_step, train_step):
    agent = agents[0]
    run_time = Timer('run').last()
    train_time = Timer('train').last()
    if run_time == 0:
        fps = 0
    else:
        fps = agent.get_env_step_intervals() / run_time
    if train_time == 0:
        tps = 0
    else:
        tps = agent.get_train_step_intervals() / train_time
    agent.store(**{
            'stats/train_step': train_step, 
            'time/fps': fps, 
            'time/tps': tps, 
        }, 
        **Timer.all_stats()
    )
    score = agent.get_raw_item('score')
    agent.store(score=score)
    agent.record(step=env_step)
    for agent in agents:
        agent.clear()



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
        time2record = agents[0].contains_stats('score') \
            and to_record(env_step)
        
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

        if time2record:
            evaluate(agents, runner, env_step, routine_config)
            save(agents)
            log(agents, env_step, train_step)


@timeit
def build_agents(config, env_stats):
    agents = []
    buffers = []
    model_name = config.model_name
    for i in range(env_stats.n_agents):
        if model_name.endswith(f'a{i}'):
            new_model_name = model_name
        else:
            new_model_name = '/'.join([model_name, f'a{i}'])
        agent_config = modify_config(
            config, 
            in_place=False, 
            aid=i, 
            seed=i*100 if config.seed is None else config.seed+i*100, 
            model_name=new_model_name, 
        )
        builder = ElementsBuilder(
            agent_config, 
            env_stats, 
            to_save_code=False, 
            max_steps=agent_config.routine['MAX_STEPS']
        )
        elements = builder.build_agent_from_scratch()
        agents.append(elements.agent)
        buffers.append(elements.buffer)
    
    return agents, buffers


def main(configs, train=train):
    config = configs[0]
    if config.routine.compute_return_at_once:
        config.buffer.sample_keys += ['advantage', 'v_target']
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    runner = Runner(config.env)

    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    # build agents
    agents, buffers = build_agents(config, env_stats)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    train(
        agents, 
        runner, 
        buffers, 
        routine_config
    )

    do_logging('Training completed')
