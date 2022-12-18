from core.typing import AttrDict, dict2AttrDict
from core.elements.model import Model


class Buffer:
    def __init__(
        self, 
        config: AttrDict,
        env_stats: AttrDict, 
        model: Model,
        aid: int=0, 
    ):
        self.config = dict2AttrDict(config, to_copy=True)
        self.aid = aid

        self.obs_keys = env_stats.obs_keys[self.aid]
        self.state_keys, self.state_type, \
            self.sample_keys, self.sample_size = \
                extract_sampling_keys(self.config, env_stats, model)

    def reset(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError


def extract_sampling_keys(
    config: AttrDict, 
    env_stats: AttrDict, 
    model: Model
):
    state_keys = model.state_keys
    state_type = model.state_type
    sample_keys = config.sample_keys
    sample_size = config.get('sample_size', config.n_steps)
    sample_keys = set(sample_keys)
    if config.get('auto_state', True):
        if state_keys is None:
            if 'state' in sample_keys:
                sample_keys.remove('state')
        else:
            sample_keys.add('state')
    obs_keys = env_stats.obs_keys[model.config.aid] if 'aid' in model.config else env_stats.obs_keys
    for k in obs_keys:
        sample_keys.add(k)
    if not config.timeout_done:
        for k in obs_keys:
            sample_keys.add(f'next_{k}')

    if env_stats.use_action_mask:
        sample_keys.append('action_mask')
    elif 'action_mask' in sample_keys:
        sample_keys.remove('action_mask')
    if env_stats.use_life_mask:
        sample_keys.append('sample_mask')
    elif 'life_mask' in sample_keys:
        sample_keys.remove('sample_mask')

    return state_keys, state_type, sample_keys, sample_size
