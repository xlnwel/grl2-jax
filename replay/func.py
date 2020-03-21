import ray

from replay.uniform import UniformReplay
from replay.per import ProportionalPER
from replay.dual import DualReplay
from replay.eps import EpisodicReplay


def create_replay(config):
    buffer_type = config['type']
    if buffer_type == 'uniform': 
        return UniformReplay(config)
    elif buffer_type == 'proportional':
        return ProportionalPER(config)
    elif buffer_type.startswith('dual'):
        return DualReplay(config)
    elif buffer_type == 'episodic':
        return EpisodicReplay(config)
    else:
        raise NotImplementedError()

def create_replay_center(config):
    RayUR = ray.remote(UniformReplay)
    RayPER = ray.remote(ProportionalPER)
    RayDR = ray.remote(DualReplay)
    RayER = ray.remote(EpisodicReplay)
    
    buffer_type = config['type']
    if buffer_type == 'uniform':
        return RayUR.remote(config)
    elif buffer_type == 'proportional':
        return RayPER.remote(config)
    elif buffer_type.startswith('dual'):
        return RayDR.remote(config)
    elif buffer_type == 'episodic':
        return RayER.remote(config)
    else:
        raise NotImplementedError()
