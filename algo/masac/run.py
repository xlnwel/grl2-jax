import collections
import numpy as np

from core.typing import tree_slice
from tools.run import RunnerWithState
from tools.utils import batch_dicts
from env.typing import EnvOutput
from algo.masac.elements.utils import concate_along_unit_dim


class Runner(RunnerWithState):
    def run(
        self, 
        n_steps, 
        agent, 
        buffer, 
        model_buffer, 
        lka_aids, 
        store_info=True,
    ):
        agent.model.switch_params(True, lka_aids)

        env_output = self.env_output
        for _ in range(n_steps):
            action, stats = agent(env_output)
            new_env_output = self.env.step(action)

            data = dict(
                obs=batch_dicts(env_output.obs, func=concate_along_unit_dim), 
                action=action, 
                reward=concate_along_unit_dim(new_env_output.reward), 
                discount=concate_along_unit_dim(new_env_output.discount), 
                next_obs=batch_dicts(self.env.prev_obs(), func=concate_along_unit_dim), 
                reset=concate_along_unit_dim(new_env_output.reset),
            )
            buffer.collect(**data, **stats)

            if model_buffer is not None:
                model_buffer.collect(
                    **data,
                    # state=stats['state'],
                )

            if store_info:
                done_env_ids = [i for i, r in enumerate(data['reset']) if np.all(r)]

                if done_env_ids:
                    info = self.env.info(done_env_ids)
                    if info:
                        info = batch_dicts(info, list)
                        agent.store(**info)
            env_output = new_env_output

        agent.model.switch_params(False, lka_aids)
        agent.model.check_params(False)

        self.env_output = env_output

    def eval_with_video(
        self, 
        agent, 
        n=None, 
        record_video=True, 
        size=(128, 128), 
        video_len=1000, 
        n_windows=4
    ):
        agent.model.check_params(False)

        if n is None:
            n = self.env.n_envs
        n_done_eps = 0
        n_run_eps = self.env.n_envs
        scores = []
        epslens = []
        frames = [collections.deque(maxlen=video_len) 
            for _ in range(min(n_windows, self.env.n_envs))]
        stats_list = []

        prev_done = np.zeros(self.env.n_envs)
        self.env.manual_reset()
        env_output = self.env.reset()
        while n_done_eps < n:
            if record_video:
                lka = self.env.get_screen(size=size)
                if self.env.env_type == 'Env':
                    frames[0].append(lka)
                else:
                    for i in range(len(frames)):
                        frames[i].append(lka[i])

            action, stats = agent(env_output)

            env_output = self.env.step(action)
            stats_list.append(stats)

            done = self.env.game_over()
            done_env_ids = [i for i, (d, pd) in 
                enumerate(zip(done, prev_done)) if d and not pd]
            n_done_eps += len(done_env_ids)

            if done_env_ids:
                score = self.env.score(done_env_ids)
                epslen = self.env.epslen(done_env_ids)
                scores += score
                epslens += epslen
                if n_run_eps < n:
                    reset_env_ids = done_env_ids[:n-n_run_eps]
                    n_run_eps += len(reset_env_ids)
                    eo = self.env.reset(reset_env_ids)
                    for t, s in zip(env_output, eo):
                        if isinstance(t, dict):
                            for k in t.keys():
                                for i, ri in enumerate(reset_env_ids):
                                    t[k][ri] = s[k][i]
                        else:
                            for i, ri in enumerate(reset_env_ids):
                                t[ri] = s[i]

        stats = batch_dicts(stats_list)
        if record_video:
            max_len = np.max([len(f) for f in frames])
            # padding to make all sequences of the same length
            for i, f in enumerate(frames):
                while len(f) < max_len:
                    f.append(f[-1])
                frames[i] = np.array(f)
            frames = np.array(frames)
            return scores, epslens, stats, frames
        else:
            return scores, epslens, stats, None


def simultaneous_rollout(env, agent, buffer, env_output, routine_config):
    agent.model.switch_params(True)
    agent.set_states()
    buffer.clear_local_buffer()
    idxes = np.arange(routine_config.n_simulated_envs)
    
    if not routine_config.switch_model_at_every_step:
        env.model.choose_elite()
    for i in range(routine_config.n_simulated_steps):
        action, stats = agent(env_output)

        env_output.obs['action'] = action
        if routine_config.switch_model_at_every_step:
            env.model.choose_elite()
        new_env_output, env_stats = env(env_output)
        env.store(**env_stats)

        data = dict(
            obs=env_output.obs, 
            action=action, 
            reward=new_env_output.reward, 
            discount=new_env_output.discount, 
            next_obs=new_env_output.obs, 
            reset=new_env_output.reset, 
            **stats
        )
        buffer.collect(idxes=idxes, **data)

        env_output = new_env_output

    agent.model.switch_params(False)
    buffer.clear_local_buffer()


def unilateral_rollout(env, agent, buffer, env_output, routine_config):
    agent.set_states()
    buffer.clear_local_buffer()
    idxes = np.arange(routine_config.n_simulated_envs)

    if not routine_config.switch_model_at_every_step:
        env.model.choose_elite()
    for aid in range(agent.env_stats.n_agents):
        lka_aids = [i for i in range(agent.env_stats.n_agents) if i != aid]
        agent.model.switch_params(True, lka_aids)

        for i in range(routine_config.n_simulated_steps):
            action, stats = agent(env_output)

            env_output.obs['action'] = action
            if routine_config.switch_model_at_every_step:
                env.model.choose_elite()
            new_env_output, env_stats = env(env_output)
            env.store(**env_stats)

            data = dict(
                obs=env_output.obs, 
                action=action, 
                reward=new_env_output.reward, 
                discount=new_env_output.discount, 
                next_obs=new_env_output.obs, 
                reset=new_env_output.reset, 
                **stats
            )
            buffer.collect(idxes=idxes, **data)
            env_output = new_env_output

        agent.model.switch_params(False, lka_aids)
        agent.model.check_params(False)
        buffer.clear_local_buffer()
    agent.model.check_params(False)

def run_on_model(env, model_buffer, agent, buffer, routine_config):
    sample_keys = buffer.obs_keys + ['state'] \
        if routine_config.restore_state else buffer.obs_keys 
    obs = model_buffer.sample_from_recency(
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
        agent.set_states(states)
    else:
        agent.set_states()

    if routine_config.lookahead_rollout == 'sim':
        return simultaneous_rollout(env, agent, buffer, env_output, routine_config)
    elif routine_config.lookahead_rollout == 'uni':
        return unilateral_rollout(env, agent, buffer, env_output, routine_config)
    else:
        raise NotImplementedError
