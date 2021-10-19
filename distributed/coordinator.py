from distributed.trainer_manager import TrainerManager
from distributed.actor_manager import ActorManager


class Coordinator:
    def __init__(self, config):
        config.trainer_manager.model = config.model
        config.actor_manager.model = config.model
        self.trainer_manager = TrainerManager(config.trainer_manager)
        self.actor_manager = ActorManager(config.actor_manager)

    def start(self):

    def allocate_worker(self, aid2eid: dict=None):
        """ We do not specify the number of workers to allocate
        so that this function immediately returns when there are
        resources available. """
        return self.worker_manager.allocate_worker.remote(aid2eid)

    def allocate_actor(self, aid2eid: dict=None):
        return self.actor_manager.allocate_worker.remote(aid2eid)
