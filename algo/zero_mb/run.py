import numpy as np

from core.typing import dict2AttrDict
from tools.display import print_dict_info
from tools.utils import batch_dicts
from env.typing import EnvOutput
from jax_tools import jax_utils


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
    model_collect, 
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

        reward = np.concatenate([eo.reward for eo in new_env_outputs], -1)
        model_collect(
            env, 0, 
            reset=np.concatenate([eo.reset for eo in new_env_outputs], -1),
            obs=batch_dicts([eo.obs for eo in env_outputs], 
                func=lambda x: np.concatenate(x, -2)),
            action=action, 
            reward=reward,
            next_obs=batch_dicts(next_obs, 
                func=lambda x: np.concatenate(x, -2))
        )

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


def split_env_output(env_output):
    env_outputs = [
        jax_utils.tree_map(lambda x: x[:, i:i+1], env_output) 
        for i in range(2)
    ]
    return env_outputs


def simultaneous_rollout(env, agents, collects, env_output, rountine_config):
    env_outputs = split_env_output(env_output)
    for agent in agents:
        agent.strategy.model.switch_params(True)
        agent.reset_states()
    
    if not rountine_config.switch_model_at_every_step:
        env.model.choose_elite()
    for i in range(rountine_config.n_imaginary_steps):
        acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

        action = np.concatenate(acts, axis=-1)
        assert action.shape == (rountine_config.n_imaginary_envs, 2), action.shape
        env_output.obs['action'] = action
        if rountine_config.switch_model_at_every_step:
            env.model.choose_elite()
        new_env_output, env_stats = env(env_output)
        new_env_outputs = split_env_output(new_env_output)
        env.store(**env_stats)

        for aid, agent in enumerate(agents):
            kwargs = dict(
                obs=env_outputs[aid].obs, 
                action=acts[aid], 
                reward=new_env_outputs[aid].reward, 
                discount=new_env_outputs[aid].discount, 
                next_obs=new_env_outputs[aid].obs, 
                **stats[aid]
            )
            collects[aid](env, 0, new_env_outputs[aid].reset, **kwargs)

        env_output = new_env_output
        env_outputs = new_env_outputs
    
    for agent in agents:
        agent.strategy.model.switch_params(False)
    return env_outputs


def unilateral_rollout(env, agents, collects, env_output, rountine_config):
    env_outputs = split_env_output(env_output)
    for aid, agent in enumerate(agents):
        for a in agents:
            a.reset_states()
        agent.strategy.model.switch_params(True)
        env.model.choose_elites()
        for i in range(rountine_config.n_imaginary_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = np.concatenate(acts, axis=-1)
            assert action.shape == (rountine_config.n_imaginary_envs, 2), action.shape
            env_output.obs['action'] = action
            new_env_output, env_stats = env(env_output)
            new_env_outputs = split_env_output(new_env_output)
            env.store(**env_stats)

            kwargs = dict(
                obs=env_outputs[aid].obs, 
                action=acts[aid], 
                reward=new_env_outputs[aid].reward, 
                discount=new_env_outputs[aid].discount, 
                next_obs=new_env_outputs[aid].obs, 
                **stats[aid]
            )
            collects[aid](env, 0, new_env_outputs[aid].reset, **kwargs)

            env_output = new_env_output
            env_outputs = new_env_outputs
        agent.strategy.model.switch_params(False)
    return env_outputs


def run_on_model(env, agents, collects, routine_config):
    obs = env.data['obs'].reshape(-1, 2, env.data['obs'].shape[-1])
    global_state = env.data['global_state'].reshape(-1, 2, env.data['global_state'].shape[-1])
    assert obs.shape[0] == global_state.shape[0], (obs.shape, global_state.shape)
    idx = np.random.randint(0, obs.shape[0], routine_config.n_imaginary_envs)
    obs = obs[idx]
    global_state = global_state[idx]
    assert len(obs.shape) == 3, obs.shape
    assert obs.shape[:2] == (routine_config.n_imaginary_envs, 2), obs.shape
    reward = np.zeros(obs.shape[:-1])
    discount = np.ones(obs.shape[:-1])
    reset = np.zeros(obs.shape[:-1])
    obs = dict2AttrDict({'obs': obs, 'global_state': global_state})

    env_output = EnvOutput(obs, reward, discount, reset)
    if routine_config.imaginary_rollout == 'sim':
        return simultaneous_rollout(env, agents, collects, env_output, routine_config)
    elif routine_config.imaginary_rollout == 'uni':
        return unilateral_rollout(env, agents, collects, env_output, routine_config)
    else:
        raise NotImplementedError
