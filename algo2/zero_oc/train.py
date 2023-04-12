import functools
import numpy as np
import ray

from core.log import do_logging
from tools.store import StateStore
from tools.timer import Every, Timer
from tools import pkg
from algo.ppo.run import *
from algo.ppo.train import main


def train(
    configs, 
    agents, 
    runner, 
    buffers, 
    routine_config
):
    def state_constructor_with_sliced_envs():
        agent_states = [a.build_memory() for a in agents]
        env_config = runner.env_config()
        env_config.n_envs //= 2
        runner_states = runner.build_env(env_config)
        return agent_states, runner_states
    
    def state_constructor():
        agent_states = [a.build_memory() for a in agents]
        runner_states = runner.build_env()
        return agent_states, runner_states
    
    def get_state():
        agent_states = [a.get_memory() for a in agents]
        runner_states = runner.get_states()
        return agent_states, runner_states
    
    def set_states(states):
        agent_states, runner_states = states
        assert len(agents) == len(agent_states)
        for a, s in zip(agents, agent_states):
            a.set_memory(s)
        runner.set_states(runner_states)

    config = configs[0]

    step = agents[0].get_env_step()
    # print("Initial running stats:", 
    #     *[f'{k:.4g}' for k in agent.get_rms_stats() if k])
    to_record = Every(
        routine_config.LOG_PERIOD, 
        start=step, 
        init_next=step != 0, 
        final=routine_config.MAX_STEPS)
    to_eval = Every(
        routine_config.EVAL_PERIOD, 
        start=step, 
        final=routine_config.MAX_STEPS)
    rt = Timer('run')
    tt = Timer('train')

    def evaluate_agent(step):
        if to_eval(step):
            eval_main = pkg.import_main('eval', config=config)
            eval_main = ray.remote(eval_main)
            p = eval_main.remote(
                configs, 
                routine_config.N_EVAL_EPISODES, 
                record=routine_config.RECORD_VIDEO, 
                fps=1, 
                info=step // routine_config.EVAL_PERIOD * routine_config.EVAL_PERIOD
            )
            return p
        else:
            return None

    for agent in agents:
        agent.store(**{'time/log_total': 0, 'time/log': 0})

    do_logging('Training starts...')
    train_step = agents[0].get_train_step()
    env_stats = runner.env_stats()
    steps_per_iter = env_stats.n_envs * routine_config.n_steps
    # eval_info = {}
    # diff_info = {}
    # with StateStore('comp', state_constructor, get_state, set_states):
    #     prev_info = run_comparisons(runner, agents)
    all_aids = list(range(len(agents)))
    while step < routine_config.MAX_STEPS:
        # train lookahead agents
        for _ in range(routine_config.n_lka_steps):
            with Timer('lookahead_run'):
                if routine_config.model_rollout_type == 'sim':
                    with StateStore('sim', 
                        state_constructor, 
                        get_state, set_states
                    ):
                        env_outputs = runner.run(
                            routine_config.n_steps, 
                            agents, buffers, 
                            all_aids, all_aids, 
                            False, compute_return=routine_config.compute_return_at_once)
                elif routine_config.model_rollout_type == 'uni':
                    env_outputs = [None for _ in all_aids]
                    for i in all_aids:
                        with StateStore(f'uni{i}', 
                            state_constructor, 
                            get_state, set_states
                        ):
                            env_outputs[i] = runner.run(
                                routine_config.n_steps, 
                                agents, buffers, 
                                [i], [i], False, 
                                compute_return=routine_config.compute_return_at_once
                            )[i]
                else:
                    raise NotImplementedError

            for agent in agents:
                with Timer('lookahead_train'):
                    agent.lookahead_train()

        # do_logging(f'start a new iteration with step: {step} vs {routine_config.MAX_STEPS}')
        start_env_step = agents[0].get_env_step()
        for i, buffer in enumerate(buffers):
            assert buffer.size() == 0, f"buffer i: {buffer.size()}"
        with rt:
            for i in all_aids:
                lka_aids = [aid for aid in all_aids if aid != i]
                with StateStore(f'real{i}', 
                    state_constructor_with_sliced_envs, 
                    get_state, set_states
                ):
                    env_outputs = runner.run(
                        routine_config.n_steps, 
                        agents, buffers, 
                        lka_aids, all_aids, 
                        compute_return=routine_config.compute_return_at_once
                    )

        for buffer in buffers:
            assert buffer.ready(), f"buffer i: ({buffer.size()}, {len(buffer._queue)})"

        step += steps_per_iter

        time2record = to_record(step)
        # if time2record:
        #     with StateStore('comp', state_constructor, get_state, set_states):
        #         before_info = run_comparisons(runner, agents)

        # train agents
        for agent in agents:
            start_train_step = agent.get_train_step()
            with tt:
                agent.train_record()
            
            train_step = agent.get_train_step()
            assert train_step != start_train_step, (start_train_step, train_step)
            agent.set_env_step(step)
            agent.trainer.sync_lookahead_params()

        # if time2record:
        #     with StateStore('comp', state_constructor, get_state, set_states):
        #         after_info = run_comparisons(runner, agents)
        # if step > 1e6:
        #     print(config.seed, np.random.randint(10000))
        #     exit()

        if time2record:
            # info = {
            #     f'diff_{k}': after_info[k] - before_info[k] 
            #     for k in before_info.keys()
            # }
            # info.update({
            #     f'dist_diff_{k}': after_info[k] - prev_info[k] 
            #     for k in before_info.keys()
            # })
            # if eval_info:
            #     eval_info = batch_dicts([eval_info, after_info], sum)
            # else:
            #     eval_info = after_info
            # if diff_info:
            #     diff_info = batch_dicts([diff_info, info], sum)
            # else:
            #     diff_info = info
            # prev_info = after_info
            eval_process = evaluate_agent(step)
            if eval_process is not None:
                with Timer('eval'):
                    scores, epslens, video = ray.get(eval_process)
                for agent in agents:
                    agent.store(**{
                        'metrics/eval_score': np.mean(scores), 
                        'metrics/eval_epslen': np.mean(epslens), 
                    })
                if video is not None:
                    agent.video_summary(video, step=step, fps=1)
            with Timer('log'):
                for agent in agents:
                    agent.store(**{
                        'stats/train_step': train_step,
                        'time/fps': (step-start_env_step)/rt.last(), 
                        'time/tps': (train_step-start_train_step)/tt.last(),
                    }, 
                    # **eval_info, **diff_info, 
                    **Timer.all_stats())
                    agent.save()
                agents[0].record(step=step)
        # do_logging(f'finish the iteration with step: {step}')


main = functools.partial(main, train=train)
