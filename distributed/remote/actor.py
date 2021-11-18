import time
import threading
import ray

from core.mixin import IdentifierConstructor
from core.mixin.monitor import create_recorder
from utility import pkg
from utility.ray_setup import config_actor
from utility.utils import dict2AttrDict
from distributed.remote.base import RayBase
from distributed.typing import ActorPair


class RemoteActorBase(RayBase):
    def __init__(self, config, env_stats, name=None):
        self.config = dict2AttrDict(config)
        self.env_stats = env_stats
        config_actor(name, self.config.coordinator.actor_config)
        self._name = name
        self._idc = IdentifierConstructor()

        self.model_constructors = {}
        self.actor_constructors = {}

        self.local_buffers = {}
        self.central_buffers = {}
        self.actors = {}
        self.configs = {}
        # we defer all constructions to the run time

        self.param_queues = {}
        self.workers = {}
        self.recorder = create_recorder(None, None)

        self._env_step = 0
        self._train_step = 0
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
        weights = self.actors[aid].actor.get_weights(identifier=identifier)
        return weights

    def get_auxiliary_weights(self, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        weights = self.actors[aid].actor.get_auxiliary_weights(identifier=identifier)
        return weights

    def construct_actor_from_config(self, config, aid=None, sid=None, weights=None):
        """ Construct an actor from config """
        config = dict2AttrDict(config)
        algo = config.algorithm
        self._setup_constructors(algo)
        actor = self._construct_actor(algo, config, self.env_stats)

        identifier = self._idc.get_identifier(aid=aid, sid=sid)
        self.actors[aid] = ActorPair(identifier, actor)
        self.configs[identifier] = config
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
        model = self.model_constructors[algo](config.model, env_stats, to_build=True)
        actor = self.actor_constructors[algo](config.actor, model)

        return actor

    def register_worker(self, wid, worker):
        self.workers[wid] = worker

    def start(self):
        self._act_thread = threading.Thread(
            target=self._act_loop, daemon=True)
        self._act_thread.start()

    def _act_loop(self):
        def wait_to_start():
            n_trainable_agents = self.env_stats['n_trainable_agents']
            n_controllable_agents = self.env_stats['n_controllable_agents']
            while len(self.local_buffers) < n_trainable_agents \
                    or len(self.param_queues) < n_trainable_agents \
                    or len(self.actors) < n_controllable_agents:
                time.sleep(1)

        wait_to_start()

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

    def fetch_weights(self, aid, q_sizes=None):
        weights = None

        if q_sizes:
            q_size = self.param_queues[aid].qsize()
            if q_size > 0:
                q_sizes.append(q_size)

        while not self.param_queues[aid].empty():
            self._train_step, (aid2, sid, weights) = self.param_queues[aid].get(block=False)
            if aid != aid2:
                raise ValueError(f'Inconsistent Agent ID: {aid} != {aid2}')

        if weights is not None:
            identifier = self._idc.get_identifier(aid=aid, sid=sid)
            self.actors[aid].actor.set_weights(weights, identifier=identifier)

        return q_sizes

    def register_param_queue(self, aid, pq):
        self.param_queues[aid] = pq

    def register_buffer(self, aid, central_buffer):
        algo = self.config.algorithm.split('-')[0]
        buffer_constructor = pkg.import_module(
            'buffer', pkg=f'distributed.{algo}').create_local_buffer
        self.local_buffers[aid] = buffer_constructor(self.config.buffer)
        self.central_buffers[aid] = central_buffer

    def _sync_sim_loop(self):
        raise NotImplementedError

    def _async_sim_loop(self):
        raise NotImplementedError

    def _send_data(self, aid):
        data = self.local_buffers[aid].sample()
        self.central_buffers[aid].merge.remote(data)
        self.local_buffers[aid].reset()


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

    actor1 = RemoteActorBase(config, env_stats, aid=1, sid=1)
    actor2 = RemoteActorBase(config, env_stats, aid=1, sid=2)

    aid1, sid1, weights1 = actor1.get_weights()
    actor2.set_weights(weights1, aid1, sid1)
    aid2, sid2, weights2 = actor2.get_weights()

    for k in weights1.keys():
        # if k.endswith('model'):
        w1 = weights1[k]
        w2 = weights2[k]
        for v1, v2 in zip(w1, w2):
            np.testing.assert_allclose(v1, v2)

    ray.shutdown()
