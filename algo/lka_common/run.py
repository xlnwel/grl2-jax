from functools import partial
import jax
import jax.numpy as jnp

from core.typing import AttrDict, dict2AttrDict
from env.typing import EnvOutput
from env.func import create_env
from tools.timer import timeit
from algo.ma_common.run import *


@timeit
def initialize_for_dynamics_run(agent, dynamics, routine_config):
    sample_keys = agent.buffer.obs_keys + ['state'] \
        if routine_config.restore_state else agent.buffer.obs_keys
    obs = dynamics.buffer.sample_from_recency(
        batch_size=routine_config.n_simulated_envs, 
        sample_keys=sample_keys, 
    )
    if obs is None:
        return None, None
    basic_shape = obs.obs.shape[:-1]
    reward = np.zeros(basic_shape, np.float32)
    discount = np.ones(basic_shape, np.float32)
    reset = np.zeros(basic_shape, np.float32)

    env_output = EnvOutput(obs, reward, discount, reset)

    if routine_config.restore_state:
        states = obs.pop('state')
        states = [states.slice(indices=uids, axis=1) 
            for uids in agent.env_stats.aid2uids]
    else:
        if agent.model.has_rnn:
            states = agent.model.get_initial_state(basic_shape[0])
        else:
            states = None

    return env_output, states


@timeit
def prepare_params(agent, dynamics):
    agent_params = agent.model.params
    dynamics_params = dynamics.model.params
    if dynamics.model.config.model_norm_obs:
        dynamics_params.obs_loc, dynamics_params.obs_scale = \
            dynamics.model.obs_rms.get_rms_stats(False)

    return agent_params, dynamics_params


@partial(jax.jit, static_argnums=[0, 2, 7])
def rollout(agent, agent_params, dynamics, dynamics_params, rng, 
            env_output, states, n_steps, elite_indices=None):
    data_list = []
    if 'sample_mask' not in env_output.obs:
        env_output.obs.sample_mask = jnp.ones_like(env_output.reset)

    for _ in jnp.arange(n_steps):
        rng, agent_rng, env_rng = jax.random.split(rng, 3)
        obs = env_output.obs
        agent_inp = []
        for aid, uids in enumerate(agent.env_stats.aid2uids):
            agent_inp.append(obs.slice(indices=uids, axis=1))
            agent_inp[-1].reset = reset = env_output.reset[:, uids]
            if agent.has_rnn:
                agent_inp[-1].state_reset = reset
                reset = jnp.expand_dims(reset, -1)
                state = states[aid]
                agent_inp[-1].state = jax.tree_util.tree_map(lambda x: x*(1-reset), state)
        action, stats, states = agent.raw_action(
            agent_params, agent_rng, agent_inp)

        model_inp = obs.copy()
        model_inp.action = action
        model_inp.reset = env_output.reset
        model_inp.obs_loc = dynamics_params.obs_loc
        model_inp.obs_scale = dynamics_params.obs_scale
        new_env_output, _, _ = dynamics.raw_action(
            dynamics_params, env_rng, model_inp, 
            # elite_indices=elite_indices
        )

        data = obs
        data.update(dict(
            action=action, 
            reward=new_env_output.reward, 
            discount=new_env_output.discount, 
            reset=new_env_output.reset, 
            **stats, 
            state=batch_dicts(states, lambda x: jnp.concatenate(x, axis=1)),
            state_reset=env_output.reset
        ))
        data.update({f'next_{k}': v for k, v in new_env_output.obs.items()})
        data_list.append(data)
        env_output = new_env_output

    return data_list, env_output, states


def concat_env_output(env_output):
    obs = batch_dicts(env_output.obs, concat_along_unit_dim)
    reward = concat_along_unit_dim(env_output.reward)
    discount = concat_along_unit_dim(env_output.discount)
    reset = concat_along_unit_dim(env_output.reset)
    return EnvOutput(obs, reward, discount, reset)


@timeit
def quantify_dynamics_errors(agent, dynamics, env_config, n_steps, lka_aids):
    dynamics.model.choose_elite(0)
    agent.model.check_params(False)
    agent.model.switch_params(True, lka_aids)

    errors = AttrDict()
    errors.trans = []
    errors.reward = []
    errors.discount = []

    env_config = env_config.copy()
    env_config.n_envs = 100
    env = create_env(env_config)
    env_output = env.output()
    env_output = concat_env_output(env_output)
    for _ in range(n_steps):
        action, _ = agent(env_output)
        new_env_output = env.step(action)
        new_env_output = concat_env_output(new_env_output)
        env_output.obs['action'] = action
        new_model_output, _ = dynamics(env_output)
        errors.trans.append(np.mean(
            np.abs(new_env_output.obs['obs'] - new_model_output.obs['obs'])))
        errors.reward.append(np.mean(
            np.abs(new_env_output.reward - new_model_output.reward)))
        errors.discount.append(np.mean(
            new_env_output.discount == new_model_output.discount))
        env_output = new_env_output

    for k, v in errors.items():
        errors[k] = np.stack(v, -1)
    agent.model.switch_params(False, lka_aids)

    return errors
