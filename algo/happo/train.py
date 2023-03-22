from algo.ma_common.train import *
from algo.lka_common.train import lka_optimize
from algo.happo.run import prepare_buffer


@timeit
def lka_env_run(agent, runner: Runner, routine_config, lka_aids, name='lka'):
    env_output = runner.run(routine_config.n_steps, agent, lka_aids, name=name)
    prepare_buffer(agent, env_output, routine_config.compute_return_at_once)


@timeit
def env_run(agent, runner: Runner, routine_config, lka_aids, name='real'):
    env_output = runner.run(routine_config.n_steps, agent, lka_aids, name=name)
    prepare_buffer(agent, env_output, routine_config.compute_return_at_once)

    env_steps_per_run = runner.get_steps_per_run(routine_config.n_steps)
    agent.add_env_step(env_steps_per_run)

    return agent.get_env_step()


@timeit
def eval_ego_and_lka(agent, runner, routine_config):
    agent.model.swap_params()
    agent.model.swap_lka_params()
    prev_ego_score, _, _ = evaluate(agent, runner, routine_config)
    lka_score, _, _ = evaluate(agent, runner, routine_config, None)
    agent.model.swap_params()
    agent.model.swap_lka_params()
    ego_score, _, _ = evaluate(agent, runner, routine_config)

    prev_ego_score = np.array(prev_ego_score)
    lka_score = np.array(lka_score)
    ego_score = np.array(ego_score)
    lka_pego_score_diff = lka_score - prev_ego_score
    ego_pego_score_diff = ego_score - prev_ego_score
    ego_lka_score_diff = ego_score - lka_score
    agent.store(
        prev_ego_score=prev_ego_score, 
        ego_score=ego_score, 
        lka_score=lka_score, 
        lka_ego_score_diff=lka_pego_score_diff, 
        ego_pego_score_diff=ego_pego_score_diff, 
        ego_lka_score_diff=ego_lka_score_diff, 
    )
    agent.model.check_current_params()
    agent.model.check_current_lka_params()


def train(
    agent, 
    runner: Runner, 
    routine_config, 
    # env_run=env_run, 
    # ego_optimize=ego_optimize
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

    while env_step < routine_config.MAX_STEPS:
        env_step = env_run(agent, runner, routine_config, lka_aids=[])
        lka_optimize(agent)
        train_step = ego_optimize(agent)
        time2record = to_record(env_step)

        if time2record:
            eval_ego_and_lka(agent, runner, routine_config)
            save(agent, None)
            log(agent, None, env_step, train_step, {})


main = partial(main, train=train)
