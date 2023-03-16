import collections
import numpy as np

from tools.run import RunnerWithState
from tools.utils import batch_dicts


def concat_along_unit_dim(x):
    x = np.concatenate(x, axis=1)
    return x


class Runner(RunnerWithState):
    def run(
        self, 
        n_steps, 
        agent, 
        lka_aids, 
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
