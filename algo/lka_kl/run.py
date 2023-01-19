import numpy as np

from tools.run import RunnerWithState
from tools.utils import batch_dicts
from env.typing import EnvOutput


def concate_along_unit_dim(x):
    x = np.concatenate(x, axis=1)
    return x


def run_eval(env, agents, img_aids, prefix=''):
    for i, agent in enumerate(agents):
        if i in img_aids:
            agent.strategy.model.switch_params(True)
        else:
            agent.strategy.model.check_params(False)

    env_output = env.output()
    np.testing.assert_allclose(env_output.reset, 1)
    env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
    infos = []
    for _ in range(env.max_episode_steps):
        acts, stats = zip(*[a(eo, evaluation=True) for a, eo in zip(agents, env_outputs)])

        action = concate_along_unit_dim(acts)
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
    np.testing.assert_allclose(env_output.reset, 1)
    for i, a in enumerate(agents):
        if prefix:
            prefix += '_'
        prefix += 'future' if i in img_aids else 'old'
    info = batch_dicts(infos, list)
    info = {f'{prefix}_{k}': np.mean(v) for k, v in info.items()}

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


class Runner(RunnerWithState):
    def run(
        self, 
        n_steps, 
        agents, 
        collects, 
        img_aids, 
        collect_ids, 
        store_info=True,
        extra_pi=False,
    ):
        assert len(img_aids) == len(agents) - 1
        for aid, agent in enumerate(agents):
            if aid in img_aids:
                agent.strategy.model.switch_params(True)
            else:
                agent.strategy.model.check_params(False)
        
        env_output = self.env_output
        env_outputs = [EnvOutput(*o) for o in zip(*env_output)]
        for _ in range(n_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])
            if extra_pi:
                for i in range(len(agents)):
                    if i not in img_aids:
                        agents[i].strategy.model.switch_params(True)
                        _, extra_stats = agents[i](env_outputs[i])
                        if "mu_logits" in extra_stats:
                            stats[i]["future_mu_logits"] = extra_stats["mu_logits"]
                        elif "mu_loc" in extra_stats:
                            stats[i]["future_mu_loc"] = extra_stats["mu_loc"]
                            stats[i]["future_mu_scale"] = extra_stats["mu_scale"]
                        stats[i]["future_mu_logprob"] = extra_stats["mu_logprob"]
                        agents[i].strategy.model.switch_params(False)
                    else:
                        if "mu_logits" in stats:
                            stats[i]["future_mu_logits"] = None
                        elif "mu_loc" in stats:
                            stats[i]["future_mu_loc"] = None
                            stats[i]["future_mu_scale"] = None
                        stats[i]["future_mu_logprob"] = None            

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
                collects[i](self.env, 0, new_env_outputs[i].reset, **kwargs)

            if store_info:
                done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if np.all(r)]

                if done_env_ids:
                    info = self.env.info(done_env_ids)
                    if info:
                        info = batch_dicts(info, list)
                        for agent in agents:
                            agent.store(**info)
            env_outputs = new_env_outputs

        for i in img_aids:
            agents[i].strategy.model.switch_params(False)
        for agent in agents:
            agent.strategy.model.check_params(False)

        self.env_output = env_output
        return env_outputs