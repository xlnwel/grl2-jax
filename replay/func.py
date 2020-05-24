import ray

from replay.uniform import UniformReplay
from replay.per import ProportionalPER
from replay.eps import EpisodicReplay
from replay.sper import SequentialPER


replay_type = dict(
    uniform=UniformReplay,
    per=ProportionalPER,
    episodic=EpisodicReplay,
    sper=SequentialPER
)

def create_replay(config, **kwargs):
    return replay_type[config['type']](config, **kwargs)

def create_replay_center(config, **kwargs):
    plain_type = replay_type[config['type']]
    ray_type = ray.remote(plain_type)
    return ray_type.remote(config, **kwargs)
    