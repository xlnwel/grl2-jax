import time
import threading

from core.mixin import IdentifierConstructor
from core.monitor import create_monitor
from utility import pkg
from utility.utils import config_attr, dict2AttrDict
from distributed.remote.base import RayBase
from distributed.typing import ActorPair


class RemoteActorBase(RayBase):
    def __init__(self, config, env_stats, name=None):
        self._env_stats = env_stats
        self.config = config_attr(self, config, filter_dict=False)
        self._name = name
        self._idc = IdentifierConstructor()

        self.model_constructors = {}
        self.actor_constructors = {}

        self.buffers = {}
        self.central_buffers = {}
        self.actors = {}
        self.configs = {}
        # we defer all constructions to the run time

        self.workers = {}
        self.monitor = create_monitor(
            self.config.root_dir, self.config.model_name, name)

        self._env_steps = 0
        self._stop_act_loop = True

    def register_actor(self, actor, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.actors[aid] = ActorPair(identifier, actor)

    def set_weights(self, weights, aid=None, sid=None, config=None):
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        if aid not in self.actors:
            config = config or self.config
            self.construct_actor_from_config(config, aid, sid, weights)
        else:
            self.actors[aid].actor.set_weights(weights, identifier=identifier)

    def get_weights(self, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.actors[aid].actor.get_weights(identifier=identifier)

    def get_auxiliary_weights(self, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.actors[aid].actor.get_auxiliary_weights(identifier=identifier)

    def construct_actor_from_config(self, config, aid=None, sid=None, weights=None):
        """ Construct an actor from config """
        algo = config.algorithm
        self._setup_constructors(algo)
        actor = self._construct_actor(algo, config, self._env_stats)

        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.actors[aid] = ActorPair(identifier, actor)
        self.configs[identifier] = dict2AttrDict(config)
        if weights is not None:
            self.actors[aid].actor.set_weights(weights, identifier=identifier)

    def _setup_constructors(self, algo):
        if algo in self.model_constructors:
            return
        self.model_constructors[algo] = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.actor_constructors[algo] = pkg.import_module(
            name='elements.actor', algo=algo, place=-1).create_actor

    def _construct_actor(self, algo, config, env_stats):
        model = self.model_constructors[algo](config.model, env_stats)
        actor = self.actor_constructors[algo](config.actor, model)

        return actor

    def register_worker(self, wid, worker):
        self.workers[wid] = worker

    def start(self):
        self._act_thread = threading.Thread(
            target=self._act_loop, daemon=True)
        self._act_thread.start()

    def _act_loop(self):
        while True:
            if self._stop_act_loop:
                time.sleep(1)
            elif self.workers == {}:
                self._sync_sim_loop()
            else:
                self._async_sim_loop()

    def stop_act_loop(self):
        self._stop_act_loop = True

    def restart_act_loop(self):
        self._stop_act_loop = False

    def fetch_weights(self, q_size):
        aid, weights = None, None

        q_size.append(self.param_queue.qsize())

        while not self._param_queue.empty():
            aid, self.train_step, weights = self._param_queue.get(block=False)

        if weights is not None:
            self.actors[aid].set_weights(weights)
        
        return q_size

    def register_buffer(self, aid, central_buffer):
        raise NotImplementedError

    def _sync_sim_loop(self):
        raise NotImplementedError

    def _async_sim_loop(self):
        raise NotImplementedError


if __name__ == '__main__':
    import numpy as np
    import ray
    from env.func import get_env_stats
    from utility.yaml_op import load_config
    config = load_config('distributed/apg/config.yaml')

    for v in config.values():
        if isinstance(v, dict):
            v['root_dir'] = config['root_dir']
            v['model_name'] = config['model_name']

    ray.init()

    env_stats = get_env_stats(config.env)

    trainer1 = RemoteActorBase(config, env_stats, aid=1, sid=1)
    trainer2 = RemoteActorBase(config, env_stats, aid=1, sid=2)

    aid1, sid1, weights1 = trainer1.get_weights()
    trainer2.set_weights(weights1, aid1, sid1)
    aid2, sid2, weights2 = trainer2.get_weights()

    for k in weights1.keys():
        # if k.endswith('model'):
        w1 = weights1[k]
        w2 = weights2[k]
        for v1, v2 in zip(w1, w2):
            np.testing.assert_allclose(v1, v2)

    ray.shutdown()
