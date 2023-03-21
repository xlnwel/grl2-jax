import os
import collections

from core.elements.builder import ElementsBuilder
from core.typing import get_basic_model_name
from tools.plot import prepare_data_for_plotting, lineplot_dataframe
from tools.store import TempStore
from tools.utils import modify_config
from replay.dual import DualReplay, PRIMAL_REPLAY
from algo.ma_common.train import *


@timeit
def build_dynamics(config, model_config, env_stats):
    root_dir = config.root_dir
    model_name = get_basic_model_name(config.model_name)
    seed = config.seed
    new_model_name = '/'.join([model_name, 'dynamics'])
    model_config = modify_config(
        model_config, 
        max_layer=1, 
        aid=0,
        algorithm=config.dynamics_name, 
        name=config.algorithm, 
        info=config.info,
        model_info=config.model_info,
        n_runners=config.env.n_runners, 
        n_envs=config.env.n_envs, 
        root_dir=root_dir, 
        model_name=new_model_name, 
        overwrite_existed_only=True, 
        seed=seed+1000
    )

    builder = ElementsBuilder(
        model_config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch(config=model_config)
    dynamics = elements.agent

    return dynamics


@timeit
def dynamics_optimize(dynamics):
    dynamics.train_record()


@timeit
def dynamics_run(agent, dynamics, routine_config, dynamics_routine_config, 
        rng, lka_aids, rollout_fn):
    if dynamics_routine_config.model_warm_up and \
        agent.get_env_step() < dynamics_routine_config.model_warm_up_steps:
        return

    def get_agent_states():
        state = agent.get_states()
        # we put the data collected from the dynamics into the secondary replay
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(routine_config.lookahead_replay)
        return state
    
    def set_agent_states(states):
        agent.set_states(states)
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(PRIMAL_REPLAY)

    # train lookahead agent
    with TempStore(get_agent_states, set_agent_states):
        rollout_fn(agent, dynamics, routine_config, rng, lka_aids)
    return True


@timeit
def lka_optimize(agent):
    agent.lookahead_train()


@timeit
def lka_train(agent, dynamics, routine_config, dynamics_routine_config, 
        n_runs, rng, lka_aids, run_fn=dynamics_run, opt_fn=lka_optimize):
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        succ = run_fn(agent, dynamics, routine_config, dynamics_routine_config, 
            rng, lka_aids)
        if succ:
            opt_fn(agent)


@timeit
def log_dynamics_errors(errors, outdir, env_step):
    if errors:
        data = collections.defaultdict(dict)
        for k1, errs in errors.items():
            for k2, v in errs.items():
                data[k2][k1] = v
        y = 'abs error'
        outdir = '/'.join([outdir, 'errors'])
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        for k, v in data.items():
            filename = f'{k}-{env_step}'
            filepath = '/'.join([outdir, filename])
            with Timer('prepare_data'):
                data[k] = prepare_data_for_plotting(
                    v, y=y, smooth_radius=0, filepath=filepath)
            # lineplot_dataframe(data[k], filename, y=y, outdir=outdir)


@timeit
def eval_ego_and_lka(agent, runner, routine_config):
    ego_score, _, _ = evaluate(agent, runner, routine_config)
    lka_optimize(agent)
    lka_score, _, _ = evaluate(agent, runner, routine_config, None)
    agent.trainer.sync_lookahead_params()
    agent.store(
        ego_score=ego_score, 
        lka_score=lka_score, 
        lka_ego_score_diff=[lka - ego for lka, ego in zip(lka_score, ego_score)]
    )


@timeit
def lka_env_run(agent, runner: Runner, routine_config, lka_aids):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    def get_fn():
        # we put the data collected from the dynamics into the secondary replay
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(routine_config.lookahead_replay)
        return get_states(agent, runner)
    
    def set_fn(states):
        set_states(states, agent, runner)
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(PRIMAL_REPLAY)

    with StateStore('lka', constructor, get_fn, set_fn):
        runner.run(routine_config.n_steps, agent, lka_aids)


@timeit
def env_run(agent, runner: Runner, routine_config, lka_aids):
    constructor = partial(state_constructor, agent=agent, runner=runner)
    get_fn = partial(get_states, agent=agent, runner=runner)
    set_fn = partial(set_states, agent=agent, runner=runner)

    with StateStore('real', constructor, get_fn, set_fn):
        runner.run(routine_config.n_steps, agent, lka_aids)

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()
