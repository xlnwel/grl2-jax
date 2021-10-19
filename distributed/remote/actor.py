
import ray

from utility import pkg
from distributed.remote.base import RayBase


class Actor(RayBase):
    def __init__(self, env_stats, aid=None):
        self._env_stats = env_stats
        self._aid = aid

        self.model_constructors = {}
        self.actor_constructors = {}

        self.actors = {}
        self.configs = {}
        # we defer all constructions to the run time

    def get_weights(self, learner):
        if getattr(self, '_normalize_obs', False):
            obs_rms = ray.get(learner.get_obs_rms_stats.remote())
            self.set_rms_stats(obs_rms=obs_rms)
        train_step, weights = ray.get(
            learner.get_train_step_weights.remote(self._pull_names))
        self.train_step = train_step
        self.model.set_weights(weights)
    
    def set_train_step_weights(self, train_step, weights):
        self.train_step = train_step
        self.model.set_weights(weights)

    def construct_actor_from_config(self, config, sid=None):
        """ Constructor the actor from config
        """
        algo = config.algorithm
        self._setup_constructors(algo)
        actor = self._construct_actor(algo, config, self._env_stats)
        
        identifier = f'{self._aid}_{algo}' \
            if self._aid is not None else algo
        if sid is not None:
            identifier = f'{identifier}_{sid}'
        self.actors[identifier] = actor
        self.configs[identifier] = config
        
    def _setup_constructors(self, algo):
        self.model_constructors[algo] = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.actor_constructors[algo] = pkg.import_module(
            name='elements.actor', algo=algo, place=-1).create_actor

    def _construct_actor(self, algo, config, env_stats):
        model = self.model_constructors[algo](config.model, env_stats)
        actor = self.actor_constructors[algo](config.actor, model)

        return actor
