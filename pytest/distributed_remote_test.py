import numpy as np
import ray

from distributed.apg.actor import create_actor
from env.func import get_env_stats
from utility.yaml_op import load_config

config = load_config('distributed/apg/config.yaml')

for v in config.values():
    if isinstance(v, dict):
        v['root_dir'] = config['root_dir']
        v['model_name'] = config['model_name']


class TestClass:
    def test_actor_init(self):
        ray.init()
        env_stats = get_env_stats(config.env)

        actor = create_actor(config, env_stats)

        ray.shutdown()

    def test_actor_sync_weights(self):
        ray.init()

        env_stats = get_env_stats(config.env)

        trainer1 = create_actor(config, env_stats, sid=1)
        trainer2 = create_actor(config, env_stats, sid=2)

        aid1, sid1, weights1 = trainer1.get_weights()
        trainer2.set_weights(weights1, aid1, sid1)
        aid2, sid2, weights2 = trainer2.get_weights()

        for k in weights1.keys():
            w1 = weights1[k]
            w2 = weights2[k]
            for v1, v2 in zip(w1, w2):
                np.testing.assert_allclose(v1, v2)

        ray.shutdown()

    # def test_actor_init(self):
    #     ray.init()
    #     env_stats = get_env_stats(config.env)

    #     trainer = create_trainer(config, env_stats)

    #     ray.shutdown()