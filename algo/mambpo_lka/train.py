from functools import partial 

from replay.dual import SECONDARY_REPLAY
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
    runner.run(MODEL_EVAL_STEPS, agent, [], collect_data=False)
    rng = agent.model.rng

    while env_step < routine_config.MAX_STEPS:
        rng, run_rng = jax.random.split(rng, 2)
        errors = AttrDict()

        lka_train(
            agent, 
            dynamics, 
            routine_config, 
            dynamics_routine_config, 
            n_runs=routine_config.n_lookahead_steps, 
            rng=run_rng, 
            lka_aids=None, 
            run_fn=dynamics_run, 
            opt_fn=lka_optimize, 
        )
        
        env_step = env_run(agent, runner, routine_config, lka_aids=None)
        time2record = to_record(env_step)
        
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
                agent, dynamics, 
                routine_config, 
                dynamics_routine_config, 
                run_rng, 
                lka_aids=None)

        train_step = ego_optimize(agent)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.ego = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            eval_ego_and_lka(agent, runner, routine_config)
            save(agent, dynamics)
            if routine_config.quantify_dynamics_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_dynamics_errors(errors, outdir, env_step)
            log(agent, dynamics, env_step, train_step, errors)

main = partial(main, train=train)
