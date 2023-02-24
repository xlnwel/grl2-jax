import numpy as np
import jax

from tools.utils import batch_dicts
from env.typing import EnvOutput
from algo.ppo.run import prepare_buffer, concate_along_unit_dim, Runner as RunnerBase


class Runner(RunnerBase):
    def run(
        self, 
        n_steps, 
        agents, 
        buffers, 
        model_buffer, 
        lka_aids, 
        collect_ids, 
        store_info=True,
        compute_return=True, 
    ):
        for aid, agent in enumerate(agents):
            if aid in lka_aids:
                agent.model.switch_params(True)
            else:
                agent.model.check_params(False)

        env_output = self.env_output
        env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
        for _ in range(n_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = concate_along_unit_dim(acts)
            new_env_output = self.env.step(action)
            new_env_outputs = [EnvOutput(*o) for o in zip(*new_env_output)]

            next_obs = self.env.prev_obs()
            for i in collect_ids:
                data = dict(
                    obs=env_outputs[i].obs, 
                    action=acts[i], 
                    reward=new_env_outputs[i].reward, 
                    discount=new_env_outputs[i].discount, 
                    next_obs=next_obs[i], 
                    reset=new_env_outputs[i].reset, 
                    **stats[i]
                )
                buffers[i].collect(**data)

            state = [s['state'] for s in stats] if 'state' in stats[0] else None
            if state is not None:
                state = batch_dicts(state, func=lambda x: np.stack(x, 1))

            if model_buffer is not None:
                model_buffer.collect(
                    reset=concate_along_unit_dim(new_env_output.reset),
                    obs=batch_dicts(env_output.obs, func=concate_along_unit_dim),
                    action=action, 
                    reward=concate_along_unit_dim(new_env_output.reward),
                    discount=concate_along_unit_dim(new_env_output.discount),
                    next_obs=batch_dicts(next_obs, func=concate_along_unit_dim), 
                    state=state,
                )

            if store_info:
                done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if np.all(r)]

                if done_env_ids:
                    info = self.env.info(done_env_ids)
                    if info:
                        info = batch_dicts(info, list)
                        for agent in agents:
                            agent.store(**info)
            env_output = new_env_output
            env_outputs = new_env_outputs

        prepare_buffer(collect_ids, agents, buffers, env_outputs, compute_return)

        for i in lka_aids:
            agents[i].model.switch_params(False)
        for agent in agents:
            agent.model.check_params(False)

        self.env_output = env_output
        return env_outputs


def split_env_output(env_output):
    env_outputs = [
        jax.tree_util.tree_map(lambda x: x[:, i:i+1], env_output) 
        for i in range(2)
    ]
    return env_outputs


def simultaneous_rollout(env, agents, buffers, env_output, rountine_config):
    env_outputs = split_env_output(env_output)
    for agent in agents:
        agent.model.switch_params(True)
        agent.set_states()
    
    if not rountine_config.switch_model_at_every_step:
        env.model.choose_elite()
    for i in range(rountine_config.n_simulated_steps):
        acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

        action = concate_along_unit_dim(acts)
        env_output.obs['action'] = action
        if rountine_config.switch_model_at_every_step:
            env.model.choose_elite()
        new_env_output, env_stats = env(env_output)
        new_env_outputs = split_env_output(new_env_output)
        env.store(**env_stats)

        for aid, agent in enumerate(agents):
            data = dict(
                obs=env_outputs[aid].obs, 
                action=acts[aid], 
                reward=new_env_outputs[aid].reward, 
                discount=new_env_outputs[aid].discount, 
                next_obs=new_env_outputs[aid].obs, 
                reset=new_env_outputs[aid].reset, 
                **stats[aid]
            )
            buffers[aid].collect(**data)

        env_output = new_env_output
        env_outputs = new_env_outputs

    prepare_buffer(list(range(len(agents))), agents, buffers, env_outputs, 
        rountine_config.compute_return_at_once)

    for agent in agents:
        agent.model.switch_params(False)

    return env_outputs


def unilateral_rollout(env, agents, buffers, env_output, rountine_config):
    env_outputs = split_env_output(env_output)
    for aid, agent in enumerate(agents):
        for a in agents:
            a.set_states()
        agent.model.switch_params(True)
        env.model.choose_elites()

        for i in range(rountine_config.n_simulated_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = concate_along_unit_dim(acts)
            assert action.shape == (rountine_config.n_simulated_envs, 2), action.shape
            env_output.obs['action'] = action
            new_env_output, env_stats = env(env_output)
            new_env_outputs = split_env_output(new_env_output)
            env.store(**env_stats)

            data = dict(
                obs=env_outputs[aid].obs, 
                action=acts[aid], 
                reward=new_env_outputs[aid].reward, 
                discount=new_env_outputs[aid].discount, 
                next_obs=new_env_outputs[aid].obs, 
                reset=new_env_outputs[aid].reset
                **stats[aid]
            )
            buffers[aid].collect(**data)

            env_output = new_env_output
            env_outputs = new_env_outputs
        agent.model.switch_params(False)

        prepare_buffer(list(range(len(agents))), agents, buffers, env_outputs, 
            rountine_config.compute_return_at_once)

    return env_outputs


def run_on_model(env, buffer, agents, buffers, routine_config):
    sample_keys = buffer.obs_keys + ['state'] \
        if routine_config.restore_state else buffer.obs_keys
    obs = buffer.sample_from_recency(
        batch_size=routine_config.n_simulated_envs,
        sample_keys=sample_keys, 
        # sample_size=1, 
        # squeeze=True, 
    )
    reward = np.zeros(obs.obs.shape[:-1])
    discount = np.ones(obs.obs.shape[:-1])
    reset = np.zeros(obs.obs.shape[:-1])

    env_output = EnvOutput(obs, reward, discount, reset)

    if routine_config.restore_state:
        states = obs.pop('state')
        states = [states.slice((slice(None), 0)), states.slice((slice(None), 1))]
        for a, s in zip(agents, states):
            a.set_states(s)
    else:
        for a in agents:
            a.set_states()

    if routine_config.lookahead_rollout == 'sim':
        return simultaneous_rollout(env, agents, buffers, env_output, routine_config)
    elif routine_config.lookahead_rollout == 'uni':
        return unilateral_rollout(env, agents, buffers, env_output, routine_config)
    else:
        raise NotImplementedError
