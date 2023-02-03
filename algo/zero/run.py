import collections
import numpy as np
import ray

from tools.run import RunnerWithState
from tools.timer import NamedEvery
from tools.utils import batch_dicts
from tools import pkg
from env.typing import EnvOutput


def concate_along_unit_dim(x):
    x = np.concatenate(x, axis=1)
    return x


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


def ray_eval(step, configs, routine_config):
    eval_main = pkg.import_main('eval', config=configs[0])
    eval_main = ray.remote(eval_main)
    p = eval_main.remote(
        configs, 
        routine_config.N_EVAL_EPISODES, 
        record=routine_config.RECORD_VIDEO, 
        fps=1, 
        info=step // routine_config.EVAL_PERIOD * routine_config.EVAL_PERIOD
    )
    return p


class Runner(RunnerWithState):
    def run(
        self, 
        n_steps, 
        agents, 
        buffers, 
        lka_aids, 
        collect_ids, 
        store_info=True, 
        compute_return=True, 
    ):
        for aid, agent in enumerate(agents):
            if aid in lka_aids:
                agent.strategy.model.switch_params(True)
            else:
                agent.strategy.model.check_params(False)
        
        env_output = self.env_output
        env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
        for _ in range(n_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = concate_along_unit_dim(acts)
            env_output = self.env.step(action)
            new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

            next_obs = self.env.prev_obs()
            for i in collect_ids:
                kwargs = dict(
                    obs=env_outputs[i].obs, 
                    action=acts[i], 
                    reward=new_env_outputs[i].reward, 
                    discount=new_env_outputs[i].discount, 
                    next_obs=next_obs[i], 
                    **stats[i]
                )
                buffers[i].collect(self.env, 0, new_env_outputs[i].reset, **kwargs)

            if store_info:
                done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if np.all(r)]

                if done_env_ids:
                    info = self.env.info(done_env_ids)
                    if info:
                        info = batch_dicts(info, list)
                        for agent in agents:
                            agent.store(**info)
            env_outputs = new_env_outputs

        prepare_buffer(collect_ids, agents, buffers, env_outputs, compute_return)

        for i in lka_aids:
            agents[i].strategy.model.switch_params(False)
        for agent in agents:
            agent.strategy.model.check_params(False)

        self.env_output = env_output
        return env_outputs

    def eval_with_video(
        self, 
        agents, 
        n=None, 
        record_video=True, 
        size=(128, 128), 
        video_len=1000, 
        n_windows=4
    ):
        for a in agents:
            a.strategy.model.check_params(False)

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
        env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
        while n_done_eps < n:
            if record_video:
                lka = self.env.get_screen(size=size)
                if self.env.env_type == 'Env':
                    frames[0].append(lka)
                else:
                    for i in range(len(frames)):
                        frames[i].append(lka[i])

            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            new_stats = {}
            for i, s in enumerate(stats):
                for k, v in s.items():
                    new_stats[f'{k}_{i}'] = v

            action = np.concatenate(acts, axis=1)
            env_output = self.env.step(action)
            env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
            for i, eo in enumerate(env_outputs):
                new_stats[f'reward_{i}'] = eo.reward
            stats_list.append(new_stats)

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

    def eval(self, agents, lka_aids, prefix=''):
        for i, agent in enumerate(agents):
            if i in lka_aids:
                agent.strategy.model.switch_params(True)
            else:
                agent.strategy.model.check_params(False)

        env_output = self.env.reset()
        np.testing.assert_allclose(env_output.reset, 1)
        env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
        infos = []
        for _ in range(self.env.max_episode_steps):
            acts, stats = zip(*[a(eo, evaluation=True) for a, eo in zip(agents, env_outputs)])

            action = concate_along_unit_dim(acts)
            env_output = self.env.step(action)
            new_env_outputs = [EnvOutput(*o) for o in zip(*env_output)]

            done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if r]

            if done_env_ids:
                info = self.env.info(done_env_ids)
                infos += info
            env_outputs = new_env_outputs

        for i in lka_aids:
            agents[i].strategy.model.switch_params(False)
        for agent in agents:
            agent.strategy.model.check_params(False)
        np.testing.assert_allclose(env_output.reset, 1)
        for i, a in enumerate(agents):
            if prefix:
                prefix += '_'
            prefix += 'future' if i in lka_aids else 'old'
        info = batch_dicts(infos, list)
        info = {f'{prefix}_{k}': np.mean(v) for k, v in info.items()}

        return info

def prepare_buffer(
    collect_ids, 
    agents, 
    buffers, 
    env_outputs, 
    compute_return=True, 
):
    for i in collect_ids:
        value = agents[i].compute_value(env_outputs[i])
        data = buffers[i].get_data({
            'value': value, 
            'state_reset': env_outputs[i].reset
        })
        if compute_return:
            value = data.value[:, :-1]
            if agents[i].trainer.config.popart:
                data.value = agents[i].trainer.popart.denormalize(data.value)
            data.value, data.next_value = data.value[:, :-1], data.value[:, 1:]
            data.advantage, data.v_target = compute_gae(
                reward=data.reward, 
                discount=data.discount,
                value=data.value,
                gamma=buffers[i].config.gamma,
                gae_discount=buffers[i].config.gamma * buffers[i].config.lam,
                next_value=data.next_value, 
                reset=data.reset,
            )
            if agents[i].trainer.config.popart:
                # reassign value to ensure value clipping at the right anchor
                data.value = value
        buffers[i].move_to_queue(data)


def compute_gae(
    reward, 
    discount, 
    value, 
    gamma,
    gae_discount, 
    next_value=None, 
    reset=None, 
):
    if next_value is None:
        value, next_value = value[:, :-1], value[:, 1:]
    elif next_value.ndim < value.ndim:
        next_value = np.expand_dims(next_value, 1)
        next_value = np.concatenate([value[:, 1:], next_value], 1)
    assert reward.shape == discount.shape == value.shape == next_value.shape, (reward.shape, discount.shape, value.shape, next_value.shape)
    
    delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    discount = (discount if reset is None else (1 - reset)) * gae_discount
    
    next_adv = 0
    advs = np.zeros_like(reward, dtype=np.float32)
    for i in reversed(range(advs.shape[1])):
        advs[:, i] = next_adv = (delta[:, i] + discount[:, i] * next_adv)
    traj_ret = advs + value

    return advs, traj_ret
