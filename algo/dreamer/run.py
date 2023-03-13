import collections
import numpy as np
import ray
import jax
import jax.numpy as jnp

from tools.run import RunnerWithState
from tools.timer import NamedEvery
from tools.utils import batch_dicts
from tools import pkg
from env.typing import RSSMEnvOutput


def concat_along_unit_dim(x):
    x = jnp.concatenate(x, axis=1)
    return x

def fake_action(basic_shape, action_dim):
    return jnp.zeros((*basic_shape, action_dim))

class Runner(RunnerWithState):
    def env_run(
        self, 
        n_steps, 
        agent, 
        buffer, 
        lka_aids, 
        store_info=True, 
    ):  
        # consider lookahead agents
        agent.strategy.model.switch_params(True, lka_aids)
        
        env_output = self.env_output
        env_outputs = [RSSMEnvOutput(*o, fake_action(o[0].obs.shape[:2], self.env_stats().action_dim[0])) for o in zip(*env_output)]
        for _ in range(n_steps):
            # acts, stats = zip(*[a(eo) for a, eo in zip(agents, env_outputs)])
            action, _ = agent(env_outputs)
            acts = []
            for aid in range(len(env_outputs)):
                acts.append(jnp.expand_dims(action[:, aid], axis=1))

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
                
            if store_info:
                done_env_ids = [i for i, r in enumerate(new_env_outputs[0].reset) if np.all(r)]

                if done_env_ids:
                    info = self.env.info(done_env_ids)
                    if info:
                        info = batch_dicts(info, list)
                        agent.store(**info)
            env_output = new_env_output
            env_outputs = new_env_outputs

        agent.strategy.model.switch_params(False, lka_aids)
        agent.strategy.model.check_params(False)

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
