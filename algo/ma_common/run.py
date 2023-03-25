from functools import partial
import collections
import numpy as np

from tools.run import RunnerWithState
from tools.store import StateStore
from tools.utils import batch_dicts


State = collections.namedtuple('state', 'agent runner')

def concat_along_unit_dim(x):
    x = np.concatenate(x, axis=1)
    return x


def state_constructor(agent, runner):
    agent_states = agent.build_memory()
    runner_states = runner.build_env()
    return State(agent_states, runner_states)


def set_states(states: State, agent, runner):
    agent_states, runner_states = states
    agent_states = agent.set_memory(agent_states)
    runner_states = runner.set_states(runner_states)
    return State(agent_states, runner_states)


class Runner(RunnerWithState):
    def run(
        self, 
        agent, 
        *, 
        name=None, 
        **kwargs, 
    ):
        if name is None:
            return self._run(agent, **kwargs)
        else:
            constructor = partial(state_constructor, agent=agent, runner=self)
            set_fn = partial(set_states, agent=agent, runner=self)
            with StateStore(name, constructor, set_fn):
                return self._run(agent, **kwargs)

    def _run(
        self, 
        agent, 
        n_steps, 
        lka_aids, 
        dynamics=None, 
        store_info=True, 
        collect_data=True, 
    ):
        agent.model.switch_params(True, lka_aids)

        env_output = self.env_output
        for _ in range(n_steps):
            action, stats = agent(env_output)
            new_env_output = self.env.step(action)

            if collect_data:
                data = dict(
                    obs=batch_dicts(env_output.obs, func=concat_along_unit_dim), 
                    action=action, 
                    reward=concat_along_unit_dim(new_env_output.reward), 
                    discount=concat_along_unit_dim(new_env_output.discount), 
                    next_obs=batch_dicts(self.env.prev_obs(), func=concat_along_unit_dim), 
                    reset=concat_along_unit_dim(new_env_output.reset),
                )
                agent.buffer.collect(**data, **stats)
                if dynamics is not None:
                    dynamics.buffer.collect(**data)

            if store_info:
                done_env_ids = [i for i, r in enumerate(new_env_output.reset[0]) if np.all(r)]

                if done_env_ids:
                    info = self.env.info(done_env_ids)
                    if info:
                        info = batch_dicts(info, list)
                        agent.store(**info)
            env_output = new_env_output

        agent.model.switch_params(False, lka_aids)
        agent.model.check_params(False)

        self.env_output = env_output

        return env_output

    def eval_with_video(
        self, 
        agent, 
        n_envs=None, 
        name=None, 
        **kwargs
    ):
        if name is None:
            return self._eval_with_video(agent, **kwargs)
        else:
            def constructor():
                env_config = self.env_config()
                if n_envs:
                    env_config.n_envs = n_envs
                agent_states = agent.build_memory()
                runner_states = self.build_env(env_config)
                return State(agent_states, runner_states)
            set_fn = partial(set_states, agent=agent, runner=self)

            with StateStore(name, constructor, set_fn):
                stats = self._eval_with_video(agent, **kwargs)
            return stats

    def _eval_with_video(
        self, 
        agent, 
        n=None, 
        record_video=True, 
        size=(128, 128), 
        video_len=1000, 
        n_windows=4
    ):
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
            prev_done = done

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
