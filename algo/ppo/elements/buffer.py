import logging

from utility.utils import dict2AttrDict
from algo.gpo.elements.buffer import LocalBufferBase, PPOBufferBase

logger = logging.getLogger(__name__)


class SamplingKeysExtractor:
    def extract_sampling_keys(self, model):
        self.state_keys = tuple([k for k in model.state_keys])
        self.state_type = model.state_type
        self.sample_keys, self.sample_size = self._get_sample_keys_size()

    def _get_sample_keys_size(self):
        state_keys = ['h', 'c']
        if self.config.get('rnn_type'): 
            sample_keys = self.config.sample_keys
            sample_size = self.config.sample_size
        else:
            sample_keys = self._remote_state_keys(
                self.config.sample_keys, 
                state_keys, 
            )
            if 'mask' in sample_keys:
                sample_keys.remove('mask')
            sample_size = None

        return sample_keys, sample_size


    def _remote_state_keys(self, sample_keys, state_keys):
        for k in state_keys:
            if k in sample_keys:
                sample_keys.remove(k)

        return sample_keys


class Sampler:
    def get_sample(
        self, 
        memory, 
        idxes, 
        sample_keys, 
    ):
        if self.state_type is None:
            sample = {k: memory[k][idxes] for k in sample_keys}
        else:
            sample = {}
            state = []
            for k in sample_keys:
                if k in self.state_keys:
                    v = memory[k][idxes, 0]
                    state.append(v.reshape(-1, v.shape[-1]))
                else:
                    sample[k] = memory[k][idxes]
            if state:
                sample['state'] = self.state_type(*state)

        return sample


class LocalBuffer(SamplingKeysExtractor, LocalBufferBase):
    pass



class PPOBuffer(Sampler, SamplingKeysExtractor, PPOBufferBase):
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
