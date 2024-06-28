import numpy as np
import ray

from distributed.remote.trainer import create_remote_trainer
from distributed.apg.actor import create_remote_actor
from envs.func import get_env_stats
from tools.yaml_op import load_config
from tools.utils import AttrDict2dict



class TestClass:
  def test_trainer_actor_sync_weights(self):
    ray.init()

    config = self._get_config()
    env_stats = get_env_stats(config.env)

    trainer = create_remote_trainer(config, env_stats, 0)
    actor = create_remote_actor(config, env_stats, 0)
    config = AttrDict2dict(config)
    ray.get(trainer.construct_strategy_from_config.remote(
      config, 0, 0))
    ray.get(actor.construct_actor_from_config.remote(
      config, 0, 0))

    weights = trainer.get_weights.remote()
    actor.set_weights.remote(weights)
    aid, sid, weights1 = ray.get(weights)
    weights2 = ray.get(actor.get_weights.remote(aid, sid))
    for k in weights1.keys():
      if k.endswith('model'):
        w1 = weights1[k]
        w2 = weights2[k]
        for v1, v2 in zip(w1, w2):
          np.testing.assert_allclose(v1, v2)

    ray.shutdown()

  def _get_config(self):
    config = load_config('distributed/apg/config.yaml')

    for v in config.values():
      if isinstance(v, dict):
        v['root_dir'] = config['root_dir']
        v['model_name'] = config['model_name']

    return config
