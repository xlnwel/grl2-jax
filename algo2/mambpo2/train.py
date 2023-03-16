from core.typing import modelpath2outdir
from tools.store import TempStore
from replay.dual import DualReplay, PRIMAL_REPLAY
from algo.masac.train import *
from algo.mambpo.run import *
from algo.happo_mb.train import log_dynamics_errors, build_dynamics


@timeit
def dynamics_optimize(dynamics):
    dynamics.train_record()


@timeit
def dynamics_run(agent, dynamics, routine_config, dynamics_routine_config, rng, lka_aids):
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

    with TempStore(get_agent_states, set_agent_states):
        if not routine_config.switch_model_at_every_step:
            dynamics.model.choose_elite()
        agent_params, dynamics_params = prepare_params(agent, dynamics)
        branched_rollout(
            agent, agent_params, dynamics, dynamics_params, 
            routine_config, rng, lka_aids)


def update_config(config, dynamics_config):
    config.buffer.primal_replay.model_norm_obs = dynamics_config.buffer.model_norm_obs


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
        env_step = env_run(agent, runner, routine_config, lka_aids=[])
        time2record = to_record(env_step)
        
        dynamics_optimize(dynamics)
        dynamics_run(
            agent, dynamics, 
            routine_config, 
            dynamics_routine_config, 
            run_rng, 
            lka_aids=[]
        )
        if routine_config.quantify_dynamics_errors and time2record:
            errors.train = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        train_step = ego_optimize(agent)
        if routine_config.quantify_dynamics_errors and time2record:
            errors.ego = quantify_dynamics_errors(
                agent, dynamics, runner.env_config(), MODEL_EVAL_STEPS, [])

        if time2record:
            evaluate(agent, dynamics, runner, env_step, routine_config)
            if routine_config.quantify_dynamics_errors:
                outdir = modelpath2outdir(agent.get_model_path())
                log_dynamics_errors(errors, outdir, env_step)
            save(agent, dynamics)
            log(agent, dynamics, env_step, train_step, errors)


def main(configs, train=train):
    config, dynamics_config = configs[0], configs[-1]
    update_config(config, dynamics_config)
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

    agent = build_agent(config, env_stats)
    dynamics = build_dynamics(config, dynamics_config, env_stats)
    dynamics.change_buffer(agent.buffer)
    save_code_for_seed(config)

    routine_config = config.routine.copy()
    dynamics_routine_config = dynamics_config.routine.copy()
    train(
        agent, 
        dynamics, 
        runner, 
        routine_config,
        dynamics_routine_config
    )

    do_logging('Training completed')
