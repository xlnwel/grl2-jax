from functools import partial

from core.log import do_logging
from tools.store import StateStore
from tools.timer import Every, Timer
from algo.ppo.run import *
from algo.ppo.train import main, train, \
    state_constructor, get_states, set_states, \
    lookahead_run, eval_and_log


def lookahead_optimize(agents, routine_config, aids):
    teammate_log_ratio = None
    for i, aid in enumerate(aids):
        agent = agents[aid]
        if i == 0:
            tlr = agent.fake_lookahead_train(teammate_log_ratio=teammate_log_ratio)
        else:
            tlr = agent.lookahead_train(teammate_log_ratio=teammate_log_ratio)
        if not routine_config.ignore_ratio_for_lookahead:
            teammate_log_ratio = tlr


def lookahead_train(agents, runner, buffers, routine_config, 
        aids, n_runs, run_fn, opt_fn):
    assert n_runs > 0, n_runs
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
        with StateStore('real', constructor, get_fn, set_fn):
            runner.run(
                routine_config.n_steps, 
                agents, buffers, 
                all_aids, all_aids, 
                compute_return=routine_config.compute_return_at_once
            )

    for buffer in buffers:
        assert buffer.ready(), f"buffer {i}: ({buffer.size()}, {len(buffer._queue)})"

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    for agent in agents:
        agent.add_env_step(env_steps_per_run)

    return agents[0].get_env_step()


def ego_optimize(agents, routine_config, aids):
    teammate_log_ratio = None

    for aid in aids:
        agent = agents[aid]
        tmp_stats = agent.train_record(teammate_log_ratio=teammate_log_ratio)
        if not routine_config.ignore_ratio_for_ego:
            teammate_log_ratio = tmp_stats["teammate_log_ratio"]

        train_step = agent.get_train_step()
        fps = agent.get_env_step_intervals() / Timer('run').last()
        tps = agent.get_train_step_intervals() / Timer('train').last()
        agent.store(**{
                'stats/train_step': train_step, 
                'time/fps': fps, 
                'time/tps': tps, 
            }, 
            **Timer.all_stats()
        )
        agent.trainer.sync_lookahead_params()
    
    return train_step


def ego_train(agents, runner, buffers, routine_config, 
        aids, run_fn, opt_fn):
    env_step = run_fn(agents, runner, buffers, routine_config)
    train_step = opt_fn(agents, routine_config, aids)

    return env_step, train_step


def train(
    agents, 
    runner, 
    buffers, 
    routine_config, 
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
    while env_step < routine_config.MAX_STEPS:
        all_aids = list(range(len(agents)))
        aids = np.random.choice(
            all_aids, size=len(all_aids), replace=False, 
            p=routine_config.perm)

        lka_train_fn(
            agents, runner, buffers, routine_config, 
            aids=aids, 
            n_runs=routine_config.n_lookahead_steps, 
            run_fn=lka_run_fn, 
            opt_fn=lka_opt_fn
        )
        env_step, _ = ego_train_fn(
            agents, runner, buffers, routine_config, 
            aids, 
            run_fn=ego_run_fn, 
            opt_fn=ego_opt_fn
        )

        time2record = agents[0].contains_stats('score') \
            and to_record(env_step)
        if time2record:
            eval_and_log(agents, runner, env_step, routine_config)

main = partial(main, train=train)
