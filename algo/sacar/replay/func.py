import ray

from algo.sacar.replay.uniform import UniformReplay
from algo.sacar.replay.per import ProportionalPER
from algo.sacar.replay.dual import DualReplay


def create_replay(config, *keys, state_shape=None):
    buffer_type = config['type']
    if buffer_type == 'uniform': 
        return UniformReplay(config, *keys, state_shape=state_shape)
    elif buffer_type == 'proportional':
        return ProportionalPER(config, *keys, state_shape=state_shape)
    elif buffer_type.startswith('dual'):
        return DualReplay(config, *keys, state_shape=state_shape)
    else:
        raise NotImplementedError()

def create_replay_center(config, *keys, state_shape=None):
    RayUP = ray.remote(UniformReplay)
    RayPER = ray.remote(ProportionalPER)
    RayDR = ray.remote(DualReplay)
    
    buffer_type = config['type']
    if buffer_type == 'uniform':
        return RayUP.remote(config, *keys, state_shape=state_shape)
    elif buffer_type == 'proportional':
        return RayPER.remote(config, *keys, state_shape=state_shape)
    elif buffer_type.startswith('dual'):
        return RayDR.remote(config, *keys, state_shape=state_shape)
    else:
        raise NotImplementedError()
