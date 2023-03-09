import jax
import jax.numpy as jnp

from core.typing import dict2AttrDict, tree_slice
from algo.masac.run import *


def rollout(env, env_params, agent, agent_params, rng, env_output, n_steps, n_envs):
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


jit_rollout = jax.jit(rollout, static_argnums=[0, 2, 6, 7])


@timeit
def simultaneous_rollout(env, agent, buffer, env_output, routine_config):
    agent.model.switch_params(True)
    agent.set_states()
    idxes = np.arange(routine_config.n_simulated_envs)
    
    if not routine_config.switch_model_at_every_step:
        env.model.choose_elite()
    env_params = env.model.params
    env_params.obs_loc, env_params.obs_scale = env.model.obs_rms.get_rms_stats(False)
    agent_params = agent.model.params
    data = jit_rollout(
        env.model, env_params, 
        agent.model, agent_params, 
        jax.random.PRNGKey(0), env_output, 
        routine_config.n_simulated_steps, 
        routine_config.n_simulated_envs)
    for d in data:
        d = [tree_slice(d, i) for i in idxes]
        buffer.merge(d)

    agent.model.switch_params(False)


@timeit
def run_on_model(env, model_buffer, agent, buffer, routine_config):
    sample_keys = buffer.obs_keys + ['state'] \
        if routine_config.restore_state else buffer.obs_keys
    obs = model_buffer.sample_from_recency(
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
        return simultaneous_rollout(env, agent, buffer, env_output, routine_config)
    elif routine_config.lookahead_rollout == 'uni':
        return unilateral_rollout(env, agent, buffer, env_output, routine_config)
    else:
        raise NotImplementedError
