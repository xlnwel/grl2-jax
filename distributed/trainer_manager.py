from distributed.remote.trainer import RemoteTrainer
from utility.utils import config_attr


class TrainerManager:
    def __init__(self, config, env_stats):
        config_attr(self, config)
        self._env_stats = env_stats

        self.trainers = {}

    def register_trainer(self, 
            trainer_id, config, aid=None, sid=None, weights=None):
        self.trainers[trainer_id] = RemoteTrainer.as_remote().remote(
            config, self._env_stats)
        self.trainers[trainer_id].construct_strategy_from_config.remote(
            config, aid=aid, sid=sid, weights=weights)

    def assiciate_actor_to_trainer(self,
            trainer_id, actor)