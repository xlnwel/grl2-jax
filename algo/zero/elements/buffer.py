import numpy as np

from utility.utils import dict2AttrDict, standardize
from algo.hm.elements.buffer import \
    LocalBufferBase, PPOBufferBase


def compute_gae(
    reward, 
    discount, 
    value, 
    value_a, 
    last_value, 
    gamma,
    gae_discount, 
    norm_adv=False, 
    mask=None, 
    epsilon=1e-8, 
    same_next_value=False
):
    if last_value is not None:
        last_value = np.expand_dims(last_value, 0)
        next_value = np.concatenate([value[1:], last_value], axis=0)
    else:
        next_value = value[1:]
        value_a = value_a[:-1]
    if same_next_value:
        next_value = np.mean(next_value, axis=-1, keepdims=True)
    assert value.ndim == next_value.ndim, (value.shape, next_value.shape)
    advs = delta = (reward + discount * gamma * next_value - value).astype(np.float32)
    next_adv = 0
    advs_a = delta = (reward + discount * gamma * next_value - value_a).astype(np.float32)
    next_adv_a = 0
    for i in reversed(range(advs.shape[0])):
        advs[i] = next_adv = (delta[i] 
            + discount[i] * gae_discount * next_adv)
        advs_a[i] = next_adv_a = (delta[i] 
            + discount[i] * gae_discount * next_adv_a)
    traj_ret = advs + value
    traj_ret_a = advs_a + value_a
    if norm_adv:
        advs = standardize(advs, mask=mask, epsilon=epsilon)

    return advs, traj_ret, traj_ret_a


class AdvantageCalculator:
    def compute_adv(self, config, last_value, data):
        if config.adv_type == 'gae':
            data['advantage'], data['traj_ret'], data['traj_ret_a'] = \
                compute_gae(
                reward=data['reward'], 
                discount=data['discount'], 
                value=data['value'], 
                value_a=data['value_a'], 
                last_value=last_value, 
                gamma=config.gamma, 
                gae_discount=config.gae_discount, 
                norm_adv=config.norm_adv == 'batch',
                mask=data.get('life_mask'),
                epsilon=config.epsilon,
                same_next_value=config.get('same_next_value', False)
            )
            data['raw_adv'] = data['advantage']
        elif config.adv_type == 'vtrace':
            pass
        else:
            raise NotImplementedError


class LocalBuffer(AdvantageCalculator, LocalBufferBase):
    pass


class PPOBuffer(AdvantageCalculator, PPOBufferBase):
    pass


def create_buffer(config, model, env_stats, **kwargs):
    config = dict2AttrDict(config)
    env_stats = dict2AttrDict(env_stats)
    BufferCls = {
        'ppo': PPOBuffer, 
        'local': LocalBuffer
    }[config.type]
    return BufferCls(
        config=config, 
        env_stats=env_stats, 
        model=model, 
        **kwargs
    )
