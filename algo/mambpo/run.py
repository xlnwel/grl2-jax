import jax
import jax.numpy as jnp

from core.typing import AttrDict, dict2AttrDict
from tools.display import print_dict_info
from tools.timer import timeit
from tools.utils import yield_from_dict
from env.typing import EnvOutput
from env.func import create_env
from algo.masac.run import *


def rollout(env, env_params, agent, agent_params, rng, env_output, n_steps):
    data_list = []

    for _ in jnp.arange(n_steps):
        rng, agent_rng, env_rng = jax.random.split(rng, 3)
        obs = dict2AttrDict(env_output.obs)
        agent_inp = [obs.slice((slice(None), uids)) 
            for uids in env.env_stats.aid2uids]
        action, stats, _ = agent.raw_action(agent_params, agent_rng, agent_inp)

        env_inp = obs.copy()
        env_inp.action = action
        env_inp.obs_loc = env_params.obs_loc
        env_inp.obs_scale = env_params.obs_scale
        new_env_output, _, _ = env.raw_action(env_params, env_rng, env_inp)

        data = obs
        data.update(dict(
            action=action, 
            reward=new_env_output.reward, 
            discount=new_env_output.discount, 
            reset=new_env_output.reset, 
            **stats
        ))
        data.update({f'next_{k}': v for k, v in new_env_output.obs.items()})
        data_list.append(data)
        env_output = new_env_output

    return data_list


jit_rollout = jax.jit(rollout, static_argnums=[0, 2, 6])


@timeit
def simultaneous_rollout(env, agent, env_output, routine_config, rng):
    agent.model.switch_params(True)
    agent.set_states()
    
    if not routine_config.switch_model_at_every_step:
        env.model.choose_elite()
    env_params = env.model.params
    env_params.obs_loc, env_params.obs_scale = env.model.obs_rms.get_rms_stats(False)
    agent_params = agent.model.params
    
    data_list = jit_rollout(
        env.model, env_params, 
        agent.model, agent_params, 
        rng, env_output, 
        routine_config.n_simulated_steps
    )
    for data in data_list:
        for d in yield_from_dict(data):
            agent.buffer.merge(d)

    agent.model.switch_params(False)


@timeit
def unilateral_rollout(env, agent, env_output, routine_config, rng):
    agent.set_states()

    if not routine_config.switch_model_at_every_step:
        env.model.choose_elite()
    env_params = env.model.params
    env_params.obs_loc, env_params.obs_scale = env.model.obs_rms.get_rms_stats(False)
    agent_params = agent.model.params
    
    for aid in range(agent.env_stats.n_agents):
        lka_aids = [i for i in range(agent.env_stats.n_agents) if i != aid]
        agent.model.switch_params(True, lka_aids)

        data_list = jit_rollout(
            env.model, env_params, 
            agent.model, agent_params, 
            rng, env_output, 
            routine_config.n_simulated_steps, 
            routine_config.n_simulated_envs)
        for data in data_list:
            for d in yield_from_dict(data):
                agent.buffer.merge(d)

        agent.model.switch_params(False, lka_aids)
        agent.model.check_params(False)


@timeit
def run_on_model(env, agent, routine_config, rng):
    sample_keys = agent.buffer.obs_keys + ['state'] \
        if routine_config.restore_state else agent.buffer.obs_keys
    obs = env.buffer.sample_from_recency(
        batch_size=routine_config.n_simulated_envs,
        sample_keys=sample_keys, 
        # sample_size=1, 
        # squeeze=True, 
    )
    if obs is None:
        return
    reward = np.zeros(obs.obs.shape[:-1])
    discount = np.ones(obs.obs.shape[:-1])
    reset = np.zeros(obs.obs.shape[:-1])

    env_output = EnvOutput(obs, reward, discount, reset)

    if routine_config.restore_state:
        states = obs.pop('state')
        states = [states.slice((slice(None), 0)), states.slice((slice(None), 1))]
        agent.set_states(states)
    else:
        agent.set_states()

    if routine_config.lookahead_rollout == 'sim':
        return simultaneous_rollout(env, agent, env_output, routine_config, rng)
    elif routine_config.lookahead_rollout == 'uni':
        return unilateral_rollout(env, agent, env_output, routine_config)
    else:
        raise NotImplementedError


def concat_env_output(env_output):
    obs = batch_dicts(env_output.obs, concat_along_unit_dim)
    reward = concat_along_unit_dim(env_output.reward)
    discount = concat_along_unit_dim(env_output.discount)
    reset = concat_along_unit_dim(env_output.reset)
    return EnvOutput(obs, reward, discount, reset)


@timeit
def quantify_model_errors(agent, model, env_config, n_steps, lka_aids):
    model.model.choose_elite(0)
    agent.model.check_params(False)
    agent.model.switch_params(True, lka_aids)

    errors = AttrDict()
    errors.trans = []
    errors.reward = []
    errors.discount = []

    env = create_env(env_config)
    env_output = env.output()
    env_output = concat_env_output(env_output)
    for _ in range(n_steps):
        action, _ = agent(env_output)
        new_env_output = env.step(action)
        new_env_output = concat_env_output(new_env_output)
        env_output.obs['action'] = action
        new_model_output, _ = model(env_output)
        errors.trans.append(
            np.abs(new_env_output.obs['obs'] - new_model_output.obs['obs']).reshape(-1))
        errors.reward.append(
            np.abs(new_env_output.reward - new_model_output.reward).reshape(-1))
        errors.discount.append(
            np.abs(new_env_output.discount - new_model_output.discount).reshape(-1))
        env_output = new_env_output

    for k, v in errors.items():
        errors[k] = np.stack(v, -1)
    agent.model.switch_params(False, lka_aids)

    return errors
