class ActorManager:
    def __init__(self) -> None:
        self.actors = None

    def set_actor(self, aid, actor):
        self.actors[aid].remote(actor)
    
    def set_actor_weights(self, aid, weights):
        self.actors[aid].set_weights.remote(weights)
