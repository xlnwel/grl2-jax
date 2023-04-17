import numpy as np
import ray

from core.elements.builder import ElementsBuilder
from core.log import do_logging
from core.typing import get_basic_model_name
from core.utils import configure_gpu, set_seed, save_code_for_seed
from tools.display import print_dict
from tools.utils import modify_config, prefix_name, flatten_dict
from tools.timer import Every, Timer, timeit
from algo.ma_common.run import Runner


@timeit
def env_run(agent, runner: Runner, routine_config, lka_aids, name='real'):
    runner.run(
        agent, 
        n_steps=routine_config.n_steps, 
        lka_aids=lka_aids, 
        name=name
    )

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


@timeit
def ego_optimize(agent, **kwargs):
    agent.train_record(**kwargs)
    train_step = agent.get_train_step()

    return train_step


@timeit
def ego_train(agent, runner, routine_config, lka_aids, run_fn, opt_fn):
    env_step = run_fn(agent, runner, routine_config, lka_aids)
    train_step = opt_fn(agent)

    return env_step, train_step


@timeit
def evaluate(agent, runner: Runner, routine_config, lka_aids=[], prev_aids=[], record_video=False, name='eval'):
    agent.model.switch_params(True, lka_aids)
    agent.model.switch_prev_params(prev_aids)

    scores, epslens, _, video = runner.eval_with_video(
        agent, 
        n_envs=routine_config.n_eval_envs, 
        record_video=record_video, 
        name=name
    )

    agent.model.switch_params(False, lka_aids)
    agent.model.switch_prev_params(prev_aids)
    agent.model.check_params(False)
    agent.model.check_current_params()

    return scores, epslens, video


@timeit
def evaluate_and_record(agent, dynamics, runner: Runner, 
                        env_step, routine_config, lka_aids=[]):
    if routine_config.EVAL_PERIOD:
        eval_scores, eval_epslens, video = evaluate(
            agent, runner, routine_config, lka_aids)
        
        agent.store(**{
            'eval_score': eval_scores, 
            'eval_epslen': eval_epslens, 
        })
        if dynamics is not None:
            dynamics.store(**{
                'dynamics_eval_score': eval_scores, 
                'dynamics_eval_epslen': eval_epslens, 
            })
        if video is not None:
            agent.video_summary(video, step=env_step, fps=1)


@timeit
def save(agent, dynamics):
    agent.save()
    if dynamics is not None: 
        dynamics.save()


@timeit
def prepare_dynamics_errors(errors):
    if not errors:
        return {}
    error_stats = {}
    for k1, errs in errors.items():
        for k2, v in errs.items():
            error_stats[f'{k1}-{k2}'] = v
    TRAIN = 'train'
    for k1, errs in errors.items():
        for k2 in errs.keys():
            if k1 != TRAIN:
                k1_err = np.mean(error_stats[f'{k1}-{k2}'])
                train_err = np.mean(error_stats[f'{TRAIN}-{k2}'])
                k1_train_err = k1_err - train_err
                error_stats[f'{k1}&{TRAIN}-{k2}'] = k1_train_err
                error_stats[f'norm_{k1}&{TRAIN}-{k2}'] = \
                    k1_train_err / train_err if train_err else k1_train_err
    error_stats = prefix_name(error_stats, 'model_error')

    return error_stats


@timeit
def log_agent(agent, env_step, train_step, error_stats):
    run_time = Timer('env_run').last()
    train_time = Timer('ego_optimize').last()
    fps = 0 if run_time == 0 else agent.get_env_step_intervals() / run_time
    tps = 0 if train_time == 0 else agent.get_train_step_intervals() / train_time
    rms = agent.actor.get_auxiliary_stats()
    rms_dict = {}
    for i, v in enumerate(rms):
        rms_dict[f'aux/obs{i}'] = v
    rms_dict[f'aux/reward'] = rms[-1]
    rms_dict = flatten_dict(rms_dict)

    agent.store(**{
            'stats/train_step': train_step, 
            'time/fps': fps, 
            'time/tps': tps, 
        }, 
        **rms_dict, 
        **error_stats, 
        **Timer.top_stats()
    )
    score = agent.get_raw_item('score')
    agent.store(score=score)
    agent.record(step=env_step)
    return score


@timeit
def log_dynamics(model, env_step, score, error_stats):
    if model is None:
        return
    train_step = model.get_train_step()
    train_time = Timer('model_train').last()
    tps = 0 if train_time == 0 else model.get_train_step_intervals() / train_time
    model.store(**{
            'stats/train_step': train_step, 
            'time/tps': tps, 
        }, 
        **error_stats, 
        **Timer.top_stats()
    )
    model.store(model_score=score)
    model.record(step=env_step)


@timeit
def log(agent, dynamics, errors):
    env_step = agent.get_env_step()
    train_step = agent.get_train_step()
    error_stats = prepare_dynamics_errors(errors)
    score = log_agent(agent, env_step, train_step, error_stats)
    log_dynamics(dynamics, env_step, score, error_stats)


@timeit
def build_agent(config, env_stats, save_monitor_stats_to_disk=True, save_config=True):
    model_name = get_basic_model_name(config.model_name)
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
    elements = builder.build_agent_from_scratch(
        save_monitor_stats_to_disk=save_monitor_stats_to_disk, 
        save_config=save_config
    )
    agent = elements.agent

    return agent


def main(configs, train):
    config = configs[0]
    seed = config.get('seed')
    set_seed(seed)

    configure_gpu()
    use_ray = config.env.get('n_runners', 1) > 1
    if use_ray:
        from tools.ray_setup import sigint_shutdown_ray
        ray.init(num_cpus=config.env.n_runners, _temp_dir="/home/ubuntu/zhanglichao/chenfeng/tmp")
        sigint_shutdown_ray()

    runner = Runner(config.env)

    env_stats = runner.env_stats()
    env_stats.n_envs = config.env.n_runners * config.env.n_envs
    print_dict(env_stats)

    agent = build_agent(config, env_stats)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    train(
        agent, 
        runner, 
        routine_config,
    )

    do_logging('Training completed')
