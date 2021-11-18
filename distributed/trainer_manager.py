from utility import pkg
from utility.utils import AttrDict2dict


class TrainerManager:
    def __init__(self, env_stats):
        self._env_stats = env_stats

        self.trainers = {}

    def get_trainer(self, trainer_id):
        return self.trainers[trainer_id]

    def register_trainer(self, 
            trainer_id, config, aid=None, sid=None, 
            weights=None, buffer=None):
        if trainer_id not in self.trainers:
            # algo = config.algorithm.split('-')[0]
            trainer_constructor = pkg.import_module(
                'trainer', pkg=f'distributed.remote').create_remote_trainer
            self.trainers[trainer_id] = trainer_constructor(
                config, self._env_stats, name=trainer_id)
        config = AttrDict2dict(config)
        self.trainers[trainer_id].construct_strategy_from_config.remote(
            config, aid=aid, sid=sid, weights=weights, buffer=buffer)
        
        return self.trainers[trainer_id]

    def assiciate_actor_to_trainer(self, trainer_id, actor):
        self.trainers[trainer_id].register_actor.remote(actor)

    def push_weights(self):
        for trainer in self.trainers.values():
            trainer.push_weights.remote()

    def register_monitor(self, monitor):
        for t in self.trainers.values():
            t.register_handler.remote(monitor=monitor)
