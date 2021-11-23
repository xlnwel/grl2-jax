from env import wrappers
from env.guandan.env import Env


def make_gd(config):
    env = Env()
    env = wrappers.post_wrap(env, config)
    
    return env