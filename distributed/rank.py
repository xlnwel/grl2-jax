import ray

from distributed.common.local.controller import Controller
from tools.ray_setup import sigint_shutdown_ray
from tools import pkg


def main(config, payoff_name, n):
  ray.init()
  sigint_shutdown_ray()

  if config.env.env_name.startswith('grl'):
    config.env.write_video = True
    config.env.dump_frequency = 1
    config.env.write_full_episode_dumps = True
    config.env.render = True
    config.runner.n_steps = config.env.max_episode_steps = 3000

  ControllerCls = pkg.import_module('local.controller', config=config).Controller
  controller: Controller = ControllerCls(config)
  controller = Controller(config, to_restore=False)
  controller.build_managers_for_evaluation(config)

  controller.evaluate_all(n, payoff_name)

  ray.shutdown()
