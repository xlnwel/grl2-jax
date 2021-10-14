import ray


class ActorManager:
    def __init__(self) -> None:
        self.actors = {}

    def set_actor(self, aid, actor):
        self.actors[aid].remote(actor)
    
    def set_actor_weights(self, aid, weights):
        self.actors[aid].set_weights.remote(weights)

    def get_auxiliary_stats(self, aid):
        return ray.get(self.actors[aid].get_auxiliary_stats.remote())
