import numpy as np

from tools.utils import batch_dicts
from env.typing import EnvOutput


def run_eval(env, agents, a1_future, a2_future, prefix):
    if a1_future:
        agents[0].strategy.model.switch_params()
        assert agents[0].strategy.model.params.imaginary == True, agents[0].strategy.model.params.imaginary
        assert agents[0].strategy.model.imaginary_params.imaginary == False, agents[0].strategy.model.imaginary_params.imaginary
    else:
        assert agents[0].strategy.model.params.imaginary == False, agents[0].strategy.model.params.imaginary
        assert agents[0].strategy.model.imaginary_params.imaginary == True, agents[0].strategy.model.imaginary_params.imaginary
    if a2_future:
        agents[1].strategy.model.switch_params()
        assert agents[1].strategy.model.params.imaginary == True, agents[1].strategy.model.params.imaginary
        assert agents[1].strategy.model.imaginary_params.imaginary == False, agents[1].strategy.model.imaginary_params.imaginary
    else:
        assert agents[1].strategy.model.params.imaginary == False, agents[1].strategy.model.params.imaginary
        assert agents[1].strategy.model.imaginary_params.imaginary == True, agents[1].strategy.model.imaginary_params.imaginary

    env_output = env.output()
    env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
    infos = []
    for _ in range(env.max_episode_steps):
        acts, stats = zip(*[a(eo, evaluation=True) for a, eo in zip(agents, env_outputs)])

        action = np.concatenate(acts, axis=-1)
        assert action.shape == (env.n_envs, len(agents)), action.shape
        env_output = env.step(action)
        new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

        done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if r]

        if done_env_ids:
            info = env.info(done_env_ids)
            infos += info
        env_outputs = new_env_outputs

    if a1_future:
        agents[0].strategy.model.switch_params()
    if a2_future:
        agents[1].strategy.model.switch_params()
    assert agents[0].strategy.model.params.imaginary == False, agents[0].strategy.model.params.imaginary
    assert agents[0].strategy.model.imaginary_params.imaginary == True, agents[0].strategy.model.imaginary_params.imaginary
    assert agents[1].strategy.model.params.imaginary == False, agents[1].strategy.model.params.imaginary
    assert agents[1].strategy.model.imaginary_params.imaginary == True, agents[1].strategy.model.imaginary_params.imaginary
    
    a1 = 'future' if a1_future else 'old'
    a2 = 'future' if a2_future else 'old'
    info = batch_dicts(infos, list)
    info = {f'{prefix}_{a1}_{a2}_{k}': v for k, v in info.items()}

    return info


def run_comparisons(env, agents, prefix):
    final_info = {}
    info = run_eval(env, agents, True, True, prefix)
    final_info.update(info)
    info = run_eval(env, agents, True, False, prefix)
    final_info.update(info)
    info = run_eval(env, agents, False, True, prefix)
    final_info.update(info)
    info = run_eval(env, agents, False, False, prefix)
    final_info.update(info)
    return final_info


def run(
    env, 
    n_steps, 
    agents, 
    collects, 
    env_outputs, 
    img_aids, 
    collect_ids, 
    store_info=True
):
    agent1_track = []
    agent2_track = []
    for i in img_aids:
        # print('run with future agents')
        agents[i].strategy.model.switch_params()
        assert agents[i].strategy.model.params.imaginary == True, agents[i].strategy.model.params.imaginary
        assert agents[i].strategy.model.imaginary_params.imaginary == False, agents[i].strategy.model.imaginary_params.imaginary
    for _ in range(n_steps):
        acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

        action = np.concatenate(acts, axis=-1)
        env_output = env.step(action)
        new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

        for i in collect_ids:
            kwargs = dict(
                obs=env_outputs[i].obs, 
                action=acts[i], 
                reward=new_env_outputs[i].reward, 
                discount=new_env_outputs[i].discount, 
                next_obs=new_env_outputs[i].obs, 
                **stats[i]
            )
            collects[i](env, 0, new_env_outputs[i].reset, **kwargs)

        if store_info:
            done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if r]

            if done_env_ids:
                info = env.info(done_env_ids)
                if info:
                    info = batch_dicts(info, list)
                    agent1_track += info.pop('agent1_track')
                    agent2_track += info.pop('agent2_track')
                    for agent in agents:
                        agent.store(**info)
        env_outputs = new_env_outputs
    for i in img_aids:
        agents[i].strategy.model.switch_params()
    for agent in agents:
        assert agent.strategy.model.params.imaginary == False, agent.strategy.model.params.imaginary
        assert agent.strategy.model.imaginary_params.imaginary == True, agent.strategy.model.imaginary_params.imaginary
    if agent1_track:
        agent1_track = sum(agent1_track)
        agent2_track = sum(agent2_track)
        return env_outputs, (agent1_track, agent2_track)
    else:
        return env_outputs, None
