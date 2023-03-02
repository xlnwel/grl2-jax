from functools import partial
import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.typing import get_basic_model_name
from core.utils import configure_gpu, set_seed, save_code_for_seed
from tools.display import print_dict
from tools.plot import prepare_data_for_plotting, lineplot_dataframe
from tools.store import StateStore, TempStore
from tools.utils import modify_config, prefix_name
from tools.timer import Every, Timer
from replay.dual import DualReplay
from .run import *
from algo.happo_mb.train import build_model


def state_constructor_with_sliced_envs(agent, runner):
    agent_states = agent.build_memory()
    env_config = runner.env_config()
    env_config.n_envs //= len(agent)
    runner_states = runner.build_env(env_config)
    return agent_states, runner_states


def state_constructor(agent, runner):
    agent_states = agent.build_memory()
    runner_states = runner.build_env()
    return agent_states, runner_states


def get_states(agent, runner):
    agent_states = agent.get_memory()
    runner_states = runner.get_states()
    return agent_states, runner_states


def set_states(states, agent, runner):
    agent_states, runner_states = states
    agent.set_memory(agent_states)
    runner.set_states(runner_states)


def model_train(model):
    if model is None:
        return
    with Timer('model_train'):
        model.train_record()


def lookahead_run(agent, model, buffer, model_buffer, routine_config):
    def get_agent_states():
        state = agent.get_states()
        # we collect lookahead data into the slow replay
        if isinstance(buffer, DualReplay):
            buffer.set_default_replay(routine_config.lookahead_replay)
        return state
    
    def set_agent_states(states):
        agent.set_states(states)
        if isinstance(buffer, DualReplay):
            buffer.set_default_replay('primal')

    # train lookahead agent
    with Timer('lookahead_run'):
        with TempStore(get_agent_states, set_agent_states):
            run_on_model(
                model, model_buffer, agent, buffer, routine_config)


def lookahead_optimize(agent):
    agent.lookahead_train()


def lookahead_train(agent, model, buffer, model_buffer, routine_config, 
        n_runs, run_fn, opt_fn):
    if model is None or not model.trainer.is_trust_worthy() \
        or not model_buffer.ready_to_sample():
        return
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        run_fn(agent, model, buffer, model_buffer, routine_config)
        opt_fn(agent)


def ego_run(agent, runner, buffer, model_buffer, routine_config):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with Timer('run'):
        with StateStore('real', constructor, get_fn, set_fn):
            runner.run(
                routine_config.n_steps, 
                agent, buffer, 
                model_buffer, 
                [], 
            )

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


def ego_optimize(agent):
    agent.train_record()
    train_step = agent.get_train_step()

    return train_step


def ego_train(agent, runner, buffer, model_buffer, routine_config, 
        run_fn, opt_fn):
    env_step = run_fn(
        agent, runner, buffer, model_buffer, routine_config)
    if buffer.ready_to_sample():
        train_step = opt_fn(agent)
    else:
        train_step = agent.get_train_step()

    return env_step, train_step


def evaluate(agent, model, runner, env_step, routine_config):
    if routine_config.EVAL_PERIOD:
        get_fn = partial(get_states, agent=agent, runner=runner)
        set_fn = partial(set_states, agent=agent, runner=runner)
        def constructor():
            env_config = runner.env_config()
            if routine_config.n_eval_envs:
                env_config.n_envs = routine_config.n_eval_envs
            agent_states = agent.build_memory()
            runner_states = runner.build_env()
            return agent_states, runner_states

        with Timer('eval'):
            with StateStore('eval', constructor, get_fn, set_fn):
                eval_scores, eval_epslens, _, video = runner.eval_with_video(
                    agent, record_video=routine_config.RECORD_VIDEO
                )
        
        agent.store(**{
            'eval_score': eval_scores, 
            'eval_epslen': eval_epslens, 
        })
        if model is not None:
            model.store(**{
                'model_eval_score': eval_scores, 
                'model_eval_epslen': eval_epslens, 
            })
        if video is not None:
            agent.video_summary(video, step=env_step, fps=1)


def save(agent, model):
    with Timer('save'):
        agent.save()
        if model is not None: 
            model.save()
            

def modelpath2outdir(model_path):
    root_dir, model_name = model_path
    model_name = get_basic_model_name(model_name)
    outdir = '/'.join([root_dir, model_name])
    return outdir


def log_model_errors(errors, outdir, env_step):
    if errors:
        with Timer('error_log'):
            data = collections.defaultdict(dict)
            for k1, errs in errors.items():
                for k2, v in errs.items():
                    data[k2][k1] = v
            y = 'abs error'
            for k, v in data.items():
                filename = f'{k}-{env_step}'
                filepath = '/'.join([outdir, 'errors', filename])
                data[k] = prepare_data_for_plotting(
                    v, y=y, smooth_radius=1, filepath=filepath)
                lineplot_dataframe(data[k], filename, y=y, outdir=outdir)


def log(agent, model, env_step, train_step, errors):
    with Timer('log'):
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
        error_stats = {}
        for k1, errs in errors.items():
            for k2, v in errs.items():
                error_stats[f'{k1}-{k2}'] = v
        TRAIN = 'train'
        for k1, errs in errors.items():
            for k2 in errs.keys():
                if k1 != TRAIN:
                    error_stats[f'{k1}&{TRAIN}-{k2}'] = error_stats[f'{k1}-{k2}'] - error_stats[f'{TRAIN}-{k2}']
        error_stats = prefix_name(error_stats, 'model_error')
        agent.store(**{
                'stats/train_step': train_step, 
                'time/fps': fps, 
                'time/tps': tps, 
            }, 
            **error_stats, 
            **Timer.all_stats()
        )
        score = agent.get_raw_item('score')
        agent.store(score=score)
        agent.record(step=env_step)

        if model is not None:
            train_step = model.get_train_step()
            model_train_duration = Timer('model_train').last()
            if model_train_duration == 0:
                tps = 0
            else:
                tps = model.get_train_step_intervals() / model_train_duration
            model.store(**{
                    'stats/train_step': train_step, 
                    'time/tps': tps, 
                }, 
                **Timer.all_stats()
            )
            model.store(model_score=score)
            model.record(step=env_step)


def train(
    agent, 
    model, 
    runner, 
    buffer, 
    model_buffer, 
    routine_config,
    model_routine_config,
    lka_run_fn=lookahead_run, 
    lka_opt_fn=lookahead_optimize, 
    lka_train_fn=lookahead_train, 
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
    model_train_fn=model_train
):
    MODEL_EVAL_STEPS = runner.env.max_episode_steps
    print('Model evaluation steps:', MODEL_EVAL_STEPS)
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )

    while env_step < routine_config.MAX_STEPS:
        errors = AttrDict()
        env_step = ego_run_fn(
            agent, runner, buffer, model_buffer, routine_config)
        time2record = agent.contains_stats('score') \
            and to_record(env_step)
        
        model_train_fn(model)
        if routine_config.quantify_model_errors and time2record:
            errors.train = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])

        if model is None or (model_routine_config.model_warm_up and env_step < model_routine_config.model_warm_up_steps):
            pass
        else:
            lka_train_fn(
                agent, 
                model, 
                buffer, 
                model_buffer, 
                routine_config, 
                n_runs=routine_config.n_lookahead_steps, 
                run_fn=lka_run_fn, 
                opt_fn=lka_opt_fn
            )

        train_step = ego_opt_fn(agent)
        if routine_config.quantify_model_errors and time2record:
            errors.ego = quantify_model_errors(
                agent, model, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            evaluate(agent, model, runner, env_step, routine_config)
            if routine_config.quantify_model_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_model_errors(errors, outdir, env_step)
            save(agent, model)
            log(agent, model, env_step, train_step, errors)


def build_agent(config, env_stats):
    model_name = config.model_name
    new_model_name = '/'.join([model_name, f'a0'])
    modify_config(
        config, 
        model_name=new_model_name, 
    )
    builder = ElementsBuilder(
        config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch()
    agent = elements.agent
    buffer = elements.buffer
    return agent, buffer


def main(configs, train=train):
    config, model_config = configs[0], configs[-1]
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners)
        sigint_shutdown_ray()

    runner = Runner(config.env)

    # load agent
    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    # build agents
    agent, buffer = build_agent(config, env_stats)
    # load model
    if config.algorithm == 'masac':
        model, model_buffer = None, None
    else:
        model, model_buffer = build_model(config, model_config, env_stats)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    model_routine_config = model_config.routine.copy()
    train(
        agent, 
        model, 
        runner, 
        buffer, 
        model_buffer, 
        routine_config,
        model_routine_config
    )

    do_logging('Training completed')
