import ray

from .remote.controller import Controller
from utility.ray_setup import sigint_shutdown_ray
from utility.utils import dict2AttrDict


def main(config):
    ray.init()
    sigint_shutdown_ray()

    configs = [dict2AttrDict(config, to_copy=True) 
        for _ in range(config.n_agents)]
    controller = Controller(configs)
    controller.pbt_train()
