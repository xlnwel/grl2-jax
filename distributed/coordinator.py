from distributed.trainer_manager import TrainerManager
from distributed.actor_manager import ActorManager
from env.func import get_env_stats
from utility.utils import config_attr


# TODO: to enable synchronous update, Coordinate need to coordinate actors and trainers
class Coordinator:
    def __init__(self, config):
        self.config = config_attr(self, config, filter_dict=True)
        env_stats = get_env_stats(config.env)

        self.trainer_manager = TrainerManager(
            config.trainer_manager, env_stats)
        self.actor_manager = ActorManager(
            config.actor_manager, env_stats)
        self.meta_strategy = MetaStrategy()

    def start(self):

    def register_trainer(self, aid2eid: dict=None):
        """ We do not specify the number of workers to allocate
        so that this function immediately returns when there are
        resources available. """
        return self.worker_manager.allocate_worker.remote(aid2eid)

    def allocate_actor(self, aid2eid: dict=None):
        return self.actor_manager.allocate_worker.remote(aid2eid)
