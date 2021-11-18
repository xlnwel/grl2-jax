import collections

from distributed.trainer_manager import TrainerManager
from distributed.actor_manager import ActorManager
from env.func import get_env_stats
from utility import pkg
from utility.utils import config_attr


# TODO: to enable synchronous update, Coordinate need to coordinate actors and trainers
class Coordinator:
    def __init__(self, config):
        self.config = config_attr(self, config, filter_dict=True)
        env_stats = get_env_stats(config.env)

        self.trainer_manager = TrainerManager(env_stats)
        self.actor_manager = ActorManager(env_stats)
        self.buffer = None
        # self.meta_strategy = MetaStrategy.remote()

    def start(self, monitor):
        tid2aids = self.register_task([0], 0, 0)
        self.register_monitor(monitor)
        self.start_task(tid2aids)

    def register_task(self, trainer_ids, aid, sid):
        tid2aids = collections.defaultdict(list)

        algo = self.config.algorithm.split('-')[0]
        buffer_constructor = pkg.import_module(
            'buffer', pkg=f'distributed.{algo}').create_central_buffer
        self.buffer = buffer_constructor(self.config.buffer)

        for trainer_id in trainer_ids:
            self.trainer_manager.register_trainer(
                trainer_id, self.config, aid, sid, buffer=self.buffer)

            for i in range(self.config.coordinator.actor_per_trainer):
                actor_id = f'{trainer_id}-{i}'
                actor = self.actor_manager.register_actor(
                    actor_id, self.config, aid, sid)
                self.trainer_manager.assiciate_actor_to_trainer(trainer_id, actor)
                tid2aids[trainer_id].append(actor_id)
        self.trainer_manager.push_weights()

        return tid2aids

    def register_monitor(self, monitor):
        self.trainer_manager.register_monitor(monitor)
        self.actor_manager.register_monitor(monitor)
        self.buffer.register_handler.remote(monitor=monitor)

    def start_task(self, tid2aids):
        for tid, aids in tid2aids.items():
            trainer = self.trainer_manager.get_trainer(tid)
            trainer.start_training.remote()
            for aid in aids:
                actor = self.actor_manager.get_actor(aid)
                actor.start.remote()
                actor.restart_act_loop.remote()


if __name__ == '__main__':
    import numpy as np
    import ray
    from utility.yaml_op import load_config
    ray.init()

    config = load_config('distributed/apg/config.yaml')

    for v in config.values():
        if isinstance(v, dict):
            v['root_dir'] = config['root_dir']
            v['model_name'] = config['model_name']


    coordinator = Coordinator(config)
    coordinator.start_trainer(0, 0, 0)

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
