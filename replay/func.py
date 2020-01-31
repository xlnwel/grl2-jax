import ray

from replay.uniform import UniformReplay
from replay.per import ProportionalPER
from replay.dual import DualReplay


def create_replay(config):
    buffer_type = config['type']
    if buffer_type == 'uniform': 
        return UniformReplay(config)
    elif buffer_type == 'proportional':
        return ProportionalPER(config)
    elif buffer_type.startswith('dual'):
        return DualReplay(config)
    else:
        raise NotImplementedError()

def create_replay_center(config):
    RayUP = ray.remote(UniformReplay)
    RayPER = ray.remote(ProportionalPER)
    RayDR = ray.remote(DualReplay)
    
    buffer_type = config['type']
    if buffer_type == 'uniform':
        return RayUP.remote(config)
    elif buffer_type == 'proportional':
        return RayPER.remote(config)
    elif buffer_type.startswith('dual'):
        return RayDR.remote(config)
    else:
        raise NotImplementedError()
