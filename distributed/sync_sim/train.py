import ray

from .remote.controller import Controller
from utility.ray_setup import sigint_shutdown_ray
from utility.utils import dict2AttrDict


def main(configs):
    ray.init()
    sigint_shutdown_ray()

    if len(configs) == 1:
        configs = [dict2AttrDict(configs[0], to_copy=True) 
            for _ in range(configs[0].n_agents)]
    controller = Controller(configs)
    controller.pbt_train()
