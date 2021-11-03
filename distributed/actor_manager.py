import ray

from utility import pkg
from utility.utils import config_attr


class ActorManager:
    def __init__(self, config, env_stats):
        self.config = config_attr(self, config, filter_dict=True)
        self._env_stats = env_stats

        self.actors = {}

    def register_actor(self, 
            actor_id, config, aid=None, sid=None, weights=None):
        if actor_id not in self.actors:
            algo = self.config.algorithm.split('-')[0]
            actor_constructor = pkg.import_module(
                'actor', pkg=f'distributed.{algo}').create_remote_actor
            self.actors[actor_id] = actor_constructor(
                config, self._env_stats, actor_id)
        self.actors[actor_id].construct_actor_from_config.remote(
            config, aid, sid, weights)

        return self.actors[actor_id]

    def set_actor_weights(self, weights, aid=None, sid=None):
        self.actors[aid].set_weights.remote(weights, aid, sid)

    def get_auxiliary_stats(self, aid=None, sid=None):
        return ray.get(self.actors[aid].get_auxiliary_stats.remote(aid, sid))
