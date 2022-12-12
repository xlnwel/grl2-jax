import numpy as np

from tools.utils import batch_dicts
from env.typing import EnvOutput


def run_eval(env, agents, img_aids, prefix=''):
    for i, agent in enumerate(agents):
        if i in img_aids:
            agent.strategy.model.switch_params(True)
        else:
            agent.strategy.model.check_params(False)

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

    for i in img_aids:
        agents[i].strategy.model.switch_params(False)
    for agent in agents:
        agent.strategy.model.check_params(False)
    
    for i, a in enumerate(agents):
        if prefix:
            prefix += '_'
        prefix += 'future' if i in img_aids else 'old'
    info = batch_dicts(infos, list)
    info = {f'{prefix}_{k}': v for k, v in info.items()}

    return info


def run_comparisons(env, agents, prefix=''):
    final_info = {}
    info = run_eval(env, agents, [0, 1], prefix)
    final_info.update(info)
    info = run_eval(env, agents, [0], prefix)
    final_info.update(info)
    info = run_eval(env, agents, [1], prefix)
    final_info.update(info)
    info = run_eval(env, agents, [], prefix)
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
    for aid, agent in enumerate(agents):
        if aid in img_aids:
            agent.strategy.model.switch_params(True)
        else:
            agent.strategy.model.check_params(False)

    for _ in range(n_steps):
        acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

        action = np.concatenate(acts, axis=-1)
        env_output = env.step(action)
        new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

        next_obs = env.prev_obs()
        for i in collect_ids:
            kwargs = dict(
                obs=env_outputs[i].obs, 
                action=acts[i], 
                reward=new_env_outputs[i].reward, 
                discount=new_env_outputs[i].discount, 
                next_obs=next_obs[i], 
                **stats[i]
            )
            collects[i](env, 0, new_env_outputs[i].reset, **kwargs)

        if store_info:
            done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if r]

            if done_env_ids:
                info = env.info(done_env_ids)
                if info:
                    info = batch_dicts(info, list)
                    for agent in agents:
                        agent.store(**info)
        env_outputs = new_env_outputs

    for i in img_aids:
        agents[i].strategy.model.switch_params(False)
    for agent in agents:
        agent.strategy.model.check_params(False)

    return env_outputs
