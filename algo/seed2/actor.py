import collections
import functools
import threading
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
import ray

from core.tf_config import *
from core.module import Ensemble
from utility.display import pwc
from utility.timer import TBTimer, Timer
from utility.utils import Every, convert_dtype
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import DataFormat, RayDataset, process_with_env
from algo.dreamer.env import make_env
from algo.dreamer.train import get_data_format


def get_learner_class(BaseAgent):
    class Learner(BaseAgent):
        """ Interface """
        def __init__(self,
                    name, 
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config):
            silence_tf_logs()
            configure_threads(4, 4)
            configure_gpu()
            configure_precision(config['precision'])
            self._dtype = global_policy().compute_dtype

            self._envs_per_worker = env_config['n_envs']
            env_config['n_envs'] = 1
            env = create_env(env_config, make_env)
            assert env.obs_dtype == np.uint8, \
                f'Expect image observation of type uint8, but get {env.obs_dtype}'
            self._action_shape = env.action_shape
            self._action_dim = env.action_dim
            self._n_ar = getattr(env, 'n_ar', 1)

            data_format = get_data_format(env, config['batch_size'], config['batch_len'])
            print(data_format)
            process = functools.partial(process_with_env, env=env, obs_range=[-.5, .5])
            dataset = RayDataset(replay, data_format, process, prefetch=20)

            self.models = Ensemble(
                model_fn=model_fn,
                config=model_config, 
                obs_shape=env.obs_shape,
                action_dim=env.action_dim, 
                is_action_discrete=env.is_action_discrete
            )

            super().__init__(
                name=name, 
                config=config, 
                models=self.models,
                dataset=dataset,
                env=env)

            self._env_step = self.global_steps.numpy()

        def distribute_weights(self, actor):
            actor.set_weights.remote(
                self.models.get_weights(name=['encoder', 'rssm', 'actor']))

        def start(self, actor):
            self.distribute_weights(actor)
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc('Learner starts learning...', color='blue')

            to_log = Every(self.LOG_PERIOD)
            while True:
                start_train_step = self.train_steps
                start_env_step = self.env_steps
                start_time = time.time()
                self.learn_log(start_env_step)
                if self.train_steps % self.SYNC_PERIOD == 0:
                    self.distribute_weights(actor)
                if to_log(self.train_steps):
                    duration = time.time() - start_time
                    self.store(
                        train_step=self.train_steps,
                        fps=(self.env_steps - start_env_step)/duration,
                        tps=(self.train_steps - start_train_step)/duration)
                    self.log(self.env_steps)
                    self.save()

    return Learner


def get_actor_class(BaseAgent):
    class Actor(BaseAgent):
        def __init__(self,
                    name,
                    model_fn,
                    config,
                    model_config,
                    env_config):
            silence_tf_logs()
            configure_threads(1, 1)
            configure_gpu()
            configure_precision(config['precision'])
            self._dtype = global_policy().compute_dtype

            self._envs_per_worker = env_config['n_envs']
            env_config['n_envs'] = config['action_batch']
            self.env = create_env(env_config, make_env)
            assert self.env.obs_dtype == np.uint8, \
                f'Expect image observation of type uint8, but get {self.env.obs_dtype}'
            self._action_shape = self.env.action_shape
            self._action_dim = self.env.action_dim
            self._n_ar = getattr(self.env, 'n_ar', 1)

            self.models = Ensemble(
                model_fn=model_fn,
                config=model_config, 
                obs_shape=self.env.obs_shape,
                action_dim=self.env.action_dim, 
                is_action_discrete=self.env.is_action_discrete
            )

            super().__init__(
                name=name, 
                config=config, 
                models=self.models,
                dataset=None,
                env=self.env)
            
            self._env_step = 0  # count the total environment steps
            # cache for episodes
            self._cache = collections.defaultdict(list)

            # agent's state
            self._state = collections.defaultdict(lambda:
                self.rssm.get_initial_state(batch_size=1, dtype=self._dtype))
            self._prev_action = collections.defaultdict(lambda:
                tf.zeros((1, self._action_dim), self._dtype))

        def set_weights(self, weights):
            self.models.set_weights(weights)

        def reset_states(self, worker_id, env_id):
            self._state[(worker_id, env_id)] = self._state.default_factory()
            self._prev_action[(worker_id, env_id)] = self._prev_action.default_factory()

        def __call__(self, worker_ids, env_ids, obs, deterministic=False):
            # pack data
            raw_state = [tf.concat(s, 0)
                for s in zip(*[tf.nest.flatten(self._state[(wid, eid)]) 
                for wid, eid in zip(worker_ids, env_ids)])]
            state_prototype = next(iter(self._state.values()))
            state = tf.nest.pack_sequence_as(
                state_prototype, raw_state)
            prev_action = tf.concat([self._prev_action[(wid, eid)] 
                for wid, eid in zip(worker_ids, env_ids)], 0)
            obs = np.stack(obs, 0)

            prev_state = state
            action, state = self.action(obs, state, prev_action, deterministic)

            prev_action = tf.one_hot(action, self._action_dim, dtype=self._dtype) \
                    if self._is_action_discrete else action
            # store states
            for wid, eid, s, a in zip(worker_ids, env_ids, zip(*state), prev_action):
                self._state[(wid, eid)] = tf.nest.pack_sequence_as(state_prototype,
                    ([tf.reshape(x, (-1, tf.shape(x)[-1])) for x in s]))
                self._prev_action[(wid, eid)] = tf.reshape(a, (-1, tf.shape(a)[-1]))
                
            if self._store_state:
                return action.numpy(), tf.nest.map_structure(lambda x: x.numpy(), prev_state)
            else:
                return action.numpy()

        def start(self, workers, replay):
            self._act_thread = threading.Thread(
                target=self._act_loop, args=[workers, replay], daemon=True)
            self._act_thread.start()
        
        def _act_loop(self, workers, replay):
            pwc('Action loop starts', color='cyan')
            objs = {workers[wid].reset_env.remote(eid): (wid, eid)
                for wid in range(self._n_workers) 
                for eid in range(self._envs_per_worker)}

            to_log = Every(self.LOG_PERIOD, self.LOG_PERIOD)
            k = 0
            start_step = self._env_step
            start_time = time.time()
            while True:
                with Timer('wait', 1000):
                    ready_objs, not_objs = ray.wait(list(objs), self._action_batch)
                with Timer('action', 1000):
                    worker_ids, env_ids = zip(*[objs[i] for i in ready_objs])
                    for oid in ready_objs:
                        del objs[oid]
                    obs, reward, discount, already_done = zip(*ray.get(ready_objs))
                    # track ready info
                    wids, eids, os, rs, ads = [], [], [], [], []
                    for wid, eid, o, r, d, ad in zip(
                        worker_ids, env_ids, obs, reward, discount, already_done):
                        if ad:
                            objs[workers[wid].reset_env.remote(eid)] = (wid, eid)
                            self.finish_episode(replay, wid, eid, o, r, d)
                            self.reset_states(wid, eid)
                        else:
                            self.store_transition(wid, eid, o, r, d)
                            wids.append(wid)
                            eids.append(eid)
                            os.append(o)
                            rs.append(r)
                            ads.append(ad)

                    if os:
                        actions = self(wids, eids, os)
                        objs.update({workers[wid].env_step.remote(eid, a): (wid, eid)
                            for wid, eid, a in zip(wids, eids, actions)})
                        [self._cache[(wid, eid)].append(dict(action=a))
                            for wid, eid, a in zip(wids, eids, actions)]
                k += 1

        def store_transition(self, worker_id, env_id, obs, reward, discount):
            if (worker_id, env_id) in self._cache:
                self._cache[(worker_id, env_id)][-1].update(dict(
                    obs=obs, 
                    reward=reward, 
                    discount=discount
                ))
            else:
                self._cache[(worker_id, env_id)].append(dict(
                    obs=obs,
                    action=np.zeros(self._action_shape, self._dtype),
                    reward=reward,
                    discount=discount
                ))

        def finish_episode(self, replay, worker_id, env_id, obs, reward, discount):
            self.store_transition(worker_id, env_id, obs, reward, discount)
            episode = self._cache.pop((worker_id, env_id))
            episode = {k: convert_dtype([t[k] for t in episode], self._precision)
                for k in episode[0]}
            replay.merge.remote(episode)
            score = np.sum(episode['reward'])
            epslen = len(episode['reward']) * self._n_ar
            self.store(score=score, epslen=epslen)
            self._env_step += epslen

    return Actor


class Worker:
    def __init__(self, name, worker_id, env_config):
        self.name = name
        self._id = worker_id
        self._n_envs = env_config['n_envs']
        env_config['n_workers'] = env_config['n_envs'] = 1
        self._envs = [create_env(env_config, make_env) 
            for _ in range(self._n_envs)]
        # self._env0_s = time.time()
        # self._env0_t = time.time()

    def reset_env(self, env_id):
        # return: obs, reward, discount, already_done
        return self._envs[env_id].reset(), 0, 1, False

    def env_step(self, env_id, action):
        # if self._id == 0 and env_id == 0:
        #     print(f'latency = {(self._env0_t-self._env0_s)*1000:.3g}ms')
        #     self._env0_s = self._env0_t
        #     self._env0_t = time.time()
        obs, reward, done, _ = self._envs[env_id].step(action)
        discount = 1 - done
        already_done = self._envs[env_id].already_done()
        return obs, reward, discount, already_done

def get_worker_class():
    return Worker
