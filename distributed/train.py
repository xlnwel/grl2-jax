import os
import ray

from core.typing import dict2AttrDict
from core.utils import configure_jax_gpu, set_seed
from tools.log import do_logging
from distributed.common.local.controller import Controller
from tools.ray_setup import init_ray
from tools.utils import modify_config
from tools import yaml_op, pkg


def save_config(config, name='config.yaml'):
  yaml_op.save_config(
    config.asdict(), 
    path=os.path.join(
      config['root_dir'], 
      config['model_name'], 
      name
    )
  )


def save_configs(configs):
  for i, config in enumerate(configs):
    save_config(config, name=f'config_a{i}.yaml')


def modify_configs(configs):
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
      c.n_agents = len(configs)

  for config in configs:
    config.parameter_server.self_play = config.self_play
    config.runner.self_play = config.self_play
    config.monitor.self_play = config.self_play
    config.runner.is_ma_algo = config.is_ma_algo
  return configs


def main(configs):
  init_ray()

  configure_jax_gpu(None)

  configs = modify_configs(configs)
  save_configs(configs)

  config = configs[0]
  seed = config.get('seed')
  set_seed(seed)

  # Construct controller
  ControllerCls = pkg.import_module('local.controller', config=config).Controller
  controller: Controller = ControllerCls(config)
  controller.build_managers(configs)
  controller.pbt_train()

  ray.shutdown()
