import ray

from core.typing import dict2AttrDict
from core.utils import configure_gpu, set_seed
from .local.controller import Controller
from tools.ray_setup import sigint_shutdown_ray
from tools.utils import modify_config, flatten_dict
from tools import yaml_op


def save_config(config, name='config.yaml'):
  yaml_op.save_config(
    config.asdict(), 
    path='/'.join([
      config['root_dir'], 
      config['model_name'], 
      name
    ])
  )

def save_configs(configs):
  for i, config in enumerate(configs):
    if config.self_play:
      config.n_agents = 2
    else:
      config.n_agents = len(configs)
    save_config(config, name=f'config_p{i}.yaml')

def main(configs):
  configure_gpu(None)
  if ray.is_initialized():
    ray.shutdown()
  ray.init()
  sigint_shutdown_ray()

  config = configs[0]
  if config.self_play:
    config.n_agents = 2
    modify_config(config, overwrite_existed_only=False, print_for_debug=False)
    configs = [config]
  elif len(configs) == 1:
    configs = [dict2AttrDict(configs[0], to_copy=True) 
      for _ in range(configs[0].n_agents)]
    for i, c in enumerate(configs):
      modify_config(c, overwrite_existed_only=True, aid=i)
      modify_config(c, overwrite_existed_only=False, print_for_debug=False)

  for config in configs:
    config.parameter_server.self_play = config.self_play
    config.runner.self_play = config.self_play
    config.monitor.self_play = config.self_play
    config.runner.is_ma_algo = config.is_ma_algo
  save_configs(configs)

  seed = config.get('seed')
  set_seed(seed)

  controller = Controller(configs[0])
  controller.build_managers(configs)
  controller.pbt_train()

  ray.shutdown()
