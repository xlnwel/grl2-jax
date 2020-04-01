import ray

from replay.uniform import UniformReplay
from replay.per import ProportionalPER
from replay.eps import EpisodicReplay


replay_type = dict(
    uniform=UniformReplay,
    proportional=ProportionalPER,
    episodic=EpisodicReplay
)

def create_replay(config):
    return replay_type[config['type']](config)

def create_replay_center(config):
    plain_type = replay_type[config['type']]
    ray_type = ray.remote(plain_type)
    return ray_type.remote(config)
    