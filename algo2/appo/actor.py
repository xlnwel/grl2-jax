import functools
import itertools
import collections
import threading
import numpy as np
import tensorflow as tf
import ray

from utility.ray_setup import cpu_affinity
from utility.utils import Every, config_attr
from utility.rl_utils import compute_act_eps
from utility.timer import Timer
from utility import pkg
from core.tf_config import *
from env.func import create_env
from env.cls import EnvOutput
from algo.ppo.buffer import compute_gae
from replay.func import create_local_buffer
from algo.seed.actor import \
    get_actor_class as get_actor_base_class, \
    get_learner_class as get_learner_base_class, \
    get_worker_class, get_evaluator_class


def get_actor_class(AgentBase):
    ActorBase = get_actor_base_class(AgentBase)
    class Actor(ActorBase):
        def set_weights(self, train_step, weights):
            self.model.set_weights(weights)
            self.train_step = train_step

    return Actor

def get_learner_class(AgentBase):
    """ A Worker is only responsible for resetting&stepping environment """
    LearnerBase = get_learner_base_class(AgentBase)
    class Learner(LearnerBase):
        def _add_attributes(self, env, dataset):
            super()._add_attributes(env, dataset)

            if not hasattr(self, '_push_names'):
                self._push_names = [k for k in self.model.keys() if 'target' not in k]
        
        def merge(self, data):
            self.buffer.merge(data)
            
        def set_handler(self, **kwargs):
            config_attr(self, kwargs)

        def _push_weights(self):
            for a in self._actors:
                a.set_weights.remote(
                    self.train_step, 
                    self.get_weights(name=self._push_names))

    return Learner
