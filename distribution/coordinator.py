from distribution.trainer_manager import TrainerManager
from distribution.actor_manager import ActorManager

class Coordinator:
    def __init__(self, config) -> None:
        self.trainer_manager = None
        self.worker_manager = None
        self.actor_manager = None

    def start(self):

    def allocate_worker(self, aid2eid: dict=None):
        """ We do not specify the number of workers to allocate
        so that this function immediately returns when there are
        resources available. """
        return self.worker_manager.allocate_worker.remote(aid2eid)

    def allocate_actor(self, aid2eid: dict=None):
        return self.actor_manager.allocate_worker.remote(aid2eid)
    