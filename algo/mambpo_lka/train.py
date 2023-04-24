from functools import partial 

from replay.dual import SECONDARY_REPLAY
from tools.timer import timeit
from algo.mambpo.train import *
from algo.lka_common.train import lka_train


@timeit
def lka_optimize(agent):
    if agent.buffer.ready_to_sample(target_replay=SECONDARY_REPLAY):
        agent.lookahead_train()


def train(
    agent, 
    dynamics, 
    runner, 
    routine_config, 
    dynamics_routine_config, 
):
    MODEL_EVAL_STEPS = runner.env.max_episode_steps
    do_logging(f'Model evaluation steps: {MODEL_EVAL_STEPS}')
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )
    runner.run(agent, n_steps=MODEL_EVAL_STEPS, lka_aids=[], collect_data=False)
    env_name = runner.env_config().env_name
    eval_data = load_eval_data(filename=env_name)
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, run_rng = jax.random.split(rng, 2)
        errors = AttrDict()

        lka_train(
            agent, 
            dynamics, 
            routine_config, 
            dynamics_routine_config, 
            n_runs=routine_config.n_lka_steps, 
            rng=run_rng, 
            lka_aids=[], 
            run_fn=dynamics_run, 
            opt_fn=lka_optimize, 
        )
        
        env_step = env_run(agent, runner, routine_config, lka_aids=None)
        time2record = to_record(env_step)
        
        if dynamics_routine_config.model_warm_up and \
            env_step < dynamics_routine_config.model_warm_up_steps:
            dynamics_optimize(dynamics, warm_up_stage=True)
        else:
            dynamics_optimize(dynamics)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.train = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if routine_config.quantify_dynamics_errors and time2record:
            errors.lka = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, None)

        if (not routine_config.use_latest_model) or \
            (dynamics_routine_config.model_warm_up and env_step < dynamics_routine_config.model_warm_up_steps):
            pass
        else:
            rng, run_rng = jax.random.split(rng, 2)
            dynamics_run(
                agent, 
                dynamics, 
                routine_config, 
                dynamics_routine_config, 
                run_rng, 
                lka_aids=None)

        if dynamics_routine_config.model_warm_up and \
            env_step < dynamics_routine_config.model_warm_up_steps:
            ego_optimize(agent, warm_up_stage=True)
        else:
            ego_optimize(agent)

        if routine_config.quantify_dynamics_errors and time2record:
            errors.ego = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            stats = dynamics.valid_stats()
            dynamics.store(**stats)
            # if eval_data:
            #     stats = dynamics.valid_stats(eval_data, 'eval')
            #     dynamics.store(**stats)
            eval_and_log(agent, dynamics, None, routine_config, 
                         agent.training_data, eval_data, eval_lka=False)

main = partial(main, train=train)
