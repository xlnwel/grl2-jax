class ActorManager:
    def __init__(self) -> None:
        self.parameter_manager = None

    def pull_weights(self, aid2eid):
        self.parameter_manager.get_weights(aid2eid)
    