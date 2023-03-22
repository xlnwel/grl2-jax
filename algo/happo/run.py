import numpy as np

from algo.ma_common.run import Runner, concat_along_unit_dim
from algo.lka_common.elements.model import LOOKAHEAD


def prepare_buffer(
    agent, 
    env_output, 
    compute_return=True, 
    lookahead=False, 
):
    buffer = agent.buffer
    value = agent.compute_value(env_output)
    data = buffer.get_data({
        'value': value, 
        'state_reset': concat_along_unit_dim(env_output.reset)
    })
    if compute_return:
        if agent.trainer.config.popart:
            value = agent.trainer.popart.denormalize(data.value)
        else:
            value = data.value
        value, next_value = value[:, :-1], value[:, 1:]
        data.advantage, data.v_target = compute_gae(
            reward=data.reward, 
            discount=data.discount,
            value=value,
            gamma=buffer.config.gamma,
            gae_discount=buffer.config.gamma * buffer.config.lam,
            next_value=next_value, 
            reset=data.reset,
        )
    data[LOOKAHEAD] = lookahead
    buffer.move_to_queue(data)


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
