from algo.masac.train import *


def mix_run(agent, model, buffer, model_buffer, routine_config):
    def get_agent_states():
        state = agent.get_states()
        # we collect lookahead data into the slow replay
        if isinstance(buffer, DualReplay):
            buffer.set_default_replay(routine_config.lookahead_replay)
        return state
    
    def set_agent_states(states):
        agent.set_states(states)
        if isinstance(buffer, DualReplay):
            buffer.set_default_replay('fast')

    # train lookahead agent
    routine_config = routine_config.copy()
    routine_config.lookahead_rollout = 'uni'
    with Timer('mix_run'):
        with TempStore(get_agent_states, set_agent_states):
            run_on_model(
                model, model_buffer, agent, buffer, routine_config)


def train(
    agent, 
    model, 
    runner, 
    buffer, 
    model_buffer, 
    routine_config, 
    lka_run_fn=lookahead_run, 
    lka_opt_fn=lookahead_optimize, 
    lka_train_fn=lookahead_train, 
    ego_run_fn=ego_run, 
    ego_opt_fn=ego_optimize, 
    ego_train_fn=ego_train, 
    model_train_fn=model_train
):
    do_logging('Training starts...')
    env_step = agent.get_env_step()
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=env_step, 
        init_next=env_step != 0, 
        final=routine_config.MAX_STEPS
    )

    while env_step < routine_config.MAX_STEPS:
        env_step = ego_run_fn(
            agent, runner, buffer, model_buffer, routine_config)
        
        model_train_fn(
            model, 
            model_buffer
        )

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
        if model_buffer.ready_to_sample():
            mix_run(agent, model, buffer, model_buffer, routine_config)
        train_step = ego_opt_fn(agent)

        time2record = agent.contains_stats('score') \
            and to_record(env_step)
        if time2record:
            eval_and_log(
                agent, model, runner, env_step, train_step, routine_config)

main = partial(main, train=train)
