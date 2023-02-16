import collections
import numpy as np
import ray
import jax

from tools.run import RunnerWithState
from tools.timer import NamedEvery
from tools.utils import batch_dicts
from tools import pkg
from env.typing import RSSMEnvOutput


def concate_along_unit_dim(x):
    x = np.concatenate(x, axis=1)
    return x


def run_model(model, buffer, routine_config):
    sample_keys = buffer.obs_keys + ['state', 'state_rssm', 'obs_rssm']
    obs = buffer.sample_from_recency(
        batch_size=routine_config.n_envs,
        sample_keys=sample_keys,
        sample_size=1,
        squeeze=True,
        n=routine_config.n_recent_trajectories
    )
    shape = obs.obs.shape[-1]

class Runner(RunnerWithState):
    
    def model_run(
        self,
    ):
        # consider sampling in model
        pass

    def env_run(
        self, 
        n_steps, 
        agents, 
        buffer, 
        lka_aids, 
        # collect_ids, 
        store_info=True, 
        # compute_return=True, 
    ):  
        # consider lookahead agents
        for aid, agent in enumerate(agents):
            if aid in lka_aids:
                agent.strategy.model.switch_params(True)
            else:
                agent.strategy.model.check_params(False)
        
        env_output = self.env_output
        env_outputs = [RSSMEnvOutput(*o, None) for o in zip(*env_output)]
        for _ in range(n_steps):
            acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])

            action = concate_along_unit_dim(acts)
            new_env_output = self.env.step(action)
            new_env_outputs = [RSSMEnvOutput(*o, jax.nn.one_hot(acts[i], self.env_stats().action_dim[0])) for i, o in enumerate(zip(*new_env_output))]

            buffer.collect(
                reset=np.concatenate(new_env_output.reset, -1),
                obs=batch_dicts(env_output.obs, 
                    func=lambda x: np.concatenate(x, -2)),
                action=action, 
                next_obs=batch_dicts(new_env_output.obs, 
                    func=lambda x: np.concatenate(x, -2)), 
                reward=np.concatenate(new_env_output.reward, -1), 
                discount=np.concatenate(new_env_output.discount, -1)
            )
            
            # next_obs = self.env.prev_obs()
            
            # for i in collect_ids:
            #     kwargs = dict(
            #         obs=env_outputs[i].obs, 
            #         action=acts[i], 
            #         reward=new_env_outputs[i].reward, 
            #         discount=new_env_outputs[i].discount, 
            #         next_obs=next_obs[i], 
            #         **stats[i]
            #     )
            #     buffers[i].collect(self.env, 0, new_env_outputs[i].reset, **kwargs)

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

        # prepare_buffer(collect_ids, agents, buffers, env_outputs, compute_return)

        for i in lka_aids:
            agents[i].strategy.model.switch_params(False)
        # for agent in agents:
            # agent.strategy.model.check_params(False)

        self.env_output = env_output
        return env_outputs

def prepare_buffer(
    collect_ids, 
    agents, 
    buffers, 
    env_outputs, 
    compute_return=True, 
):
    for i in collect_ids:
        env_outputs
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
