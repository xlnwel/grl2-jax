import numpy as np
import ray

from distributed.coordinator import Coordinator
from distributed.remote.trainer import create_remote_trainer
from env.func import get_env_stats
from tools.yaml_op import load_config
from tools.utils import AttrDict2dict


class TestClass:
  # def test_trainer_sync_weights(self):
  #   ray.init()

  #   config = self._get_config()
  #   env_stats = get_env_stats(config.env)

  #   trainer1 = create_remote_trainer(config, env_stats, 0)
  #   trainer2 = create_remote_trainer(config, env_stats, 1)
  #   config = AttrDict2dict(config)
  #   ray.get(trainer1.construct_strategy_from_config.remote(
  #     config, 0, 0))
  #   ray.get(trainer2.construct_strategy_from_config.remote(
  #     config, 0, 0))

  #   weights = trainer1.get_weights.remote()
  #   trainer2.set_weights.remote(weights)
  #   aid2, sid2, weights2 = ray.get(trainer2.get_weights.remote())
  #   aid1, sid1, weights1 = ray.get(weights)
  #   for k in weights1.keys():
  #     # if k.endswith('model'):
  #     w1 = weights1[k]
  #     w2 = weights2[k]
  #     for v1, v2 in zip(w1, w2):
  #       np.testing.assert_allclose(v1, v2)

  #   ray.shutdown()

  def test_coordinator_trainer_sync_actors(self):
    ray.init()

    config = self._get_config()

    coordinator = Coordinator(config)
    coordinator.register_task([0], 0, 0)

    trainer = coordinator.trainer_manager.trainers[0]
    aid, sid, w = ray.get(trainer.get_weights.remote())
    ray.get(trainer.push_weights.remote())
    for a in coordinator.actor_manager.actors.values():
      a.fetch_weights.remote(aid)
      w2 = ray.get(a.get_weights.remote(aid, sid))
      for k in w.keys():
        if k.endswith('model'):
          for v1, v2 in zip(w[k], w2[k]):
            np.testing.assert_allclose(v1, v2)

    ray.shutdown()

  def test_coordinator_trainer_sync_actors(self):
    ray.init()

    config = self._get_config()

    coordinator = Coordinator(config)
    coordinator.start()

    ray.shutdown()

  def _get_config(self):
    config = load_config('distributed/apg/config.yaml')

    for v in config.values():
      if isinstance(v, dict):
        v['root_dir'] = config['root_dir']
        v['model_name'] = config['model_name']
    
    return config
