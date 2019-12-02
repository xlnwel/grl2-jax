import ray

from replay.uniform import UniformReplay
from replay.per import ProportionalPER


def create_replay(config, *keys, state_shape=None):
    buffer_type = config['type']
    if buffer_type == 'uniform': 
        return UniformReplay(config, *keys, state_shape=state_shape)
    elif buffer_type == 'proportional':
        return ProportionalPER(config, *keys, state_shape=state_shape)
    else:
        raise NotImplementedError()

def create_replay_center(config, *keys, state_shape=None):
    RayUP = ray.remote(UniformReplay)
    RayPER = ray.remote(ProportionalPER)
    buffer_type = config['type']
    if buffer_type == 'uniform':
        return RayUP.remote(config, *keys, state_shape=state_shape)
    elif buffer_type == 'proportional':
        return RayPER.remote(config, *keys, state_shape=state_shape)
    else:
        raise NotImplementedError()
