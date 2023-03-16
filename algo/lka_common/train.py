import os
import collections

from core.elements.builder import ElementsBuilder
from core.typing import get_basic_model_name
from tools.plot import prepare_data_for_plotting, lineplot_dataframe
from tools.store import TempStore
from tools.utils import modify_config
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
def prepare_params(agent, dynamisc):
    agent_params = agent.model.params
    dynamics_params = dynamisc.model.params
    dynamics_params.obs_loc, dynamics_params.obs_scale = \
        dynamisc.model.obs_rms.get_rms_stats(False)

    return agent_params, dynamics_params


@timeit
def dynamics_run(agent, dynamics, routine_config, dynamics_routine_config, 
        rng, lka_aids, rollout_fn):
    if dynamics_routine_config.model_warm_up and \
        agent.get_env_step() < dynamics_routine_config.model_warm_up_steps:
        return

    def get_agent_states():
        state = agent.get_states()
        return state
    
    def set_agent_states(state):
        agent.set_states(state)

    # train lookahead agent
    with TempStore(get_agent_states, set_agent_states):
        agent_params, dynamics_params = prepare_params(agent, dynamics)
        rollout_fn(
            agent, agent_params, 
            dynamics, dynamics_params, 
            routine_config, rng, lka_aids
        )


@timeit
def lka_optimize(agent):
    agent.lookahead_train()


@timeit
def lka_train(agent, dynamics, routine_config, dynamics_routine_config, 
        n_runs, rng, lka_aids, run_fn, opt_fn):
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        run_fn(agent, dynamics, routine_config, dynamics_routine_config, 
            rng, lka_aids)
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
