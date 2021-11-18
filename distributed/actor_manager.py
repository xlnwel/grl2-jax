import ray

from utility import pkg
from utility.utils import AttrDict2dict


class ActorManager:
    def __init__(self, env_stats):
        self._env_stats = env_stats

        self.actors = {}

    def get_actor(self, actor_id):
        return self.actors[actor_id]

    def register_actor(self, 
            actor_id, config, aid=None, sid=None, weights=None):
        if actor_id not in self.actors:
            algo = config.algorithm.split('-')[0]
            actor_constructor = pkg.import_module(
                'actor', pkg=f'distributed.{algo}').create_remote_actor
            self.actors[actor_id] = actor_constructor(
                config, self._env_stats, name=actor_id)
        config = AttrDict2dict(config)
        self.actors[actor_id].construct_actor_from_config.remote(
            config, aid, sid, weights)

        return self.actors[actor_id]

    def set_actor_weights(self, 
            weights, actor_id, aid=None, sid=None):
        self.actors[actor_id].set_weights.remote(weights, aid, sid)

    def get_auxiliary_stats(self, 
            actor_id, aid=None, sid=None):
        return ray.get(
            self.actors[actor_id].get_auxiliary_stats.remote(aid, sid))

    def register_monitor(self, monitor):
        for a in self.actors.values():
            a.register_handler.remote(monitor=monitor)
