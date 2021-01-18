import collections
import functools
import threading
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy
import ray

from core.tf_config import *
from core.dataset import create_dataset
from utility.display import pwc
from utility.timer import TBTimer
from utility.utils import Every, convert_dtype
from utility.ray_setup import cpu_affinity, get_num_cpus
from utility import pkg
from env.func import create_env
from replay.func import create_replay
from algo.dreamer.env import make_env
from algo.apex.actor import config_actor, get_base_learner_class


def get_learner_class(BaseAgent):
    BaseLearner = get_base_learner_class(BaseAgent)
    class Learner(BaseLearner):
        """ Interface """
        def __init__(self,
                    name, 
                    model_fn,
                    config, 
                    model_config,
                    env_config,
                    replay_config):
            config_actor('Learner', config)

            self._envs_per_worker = env_config['n_envs']
            env = create_env(env_config)
            
            models = model_fn(model_config, env)
            
            if getattr(models, 'state_size', None):
                replay_config['state_keys'] = list(models.state_keys)
            self.replay = create_replay(replay_config)
            self.replay.load_data()

            am = pkg.import_module('agent', config=config)
            data_format = am.get_data_format(
                env=env, replay_config=replay_config, 
                agent_config=config, model=models)
            dataset = create_dataset(self.replay, env, data_format=data_format)

            super().__init__(
                name=name, 
                config=config, 
                models=models,
                dataset=dataset,
                env=env)

            self.env_step = self._env_step.numpy()  # count the total environment steps
            # cache for episodes
            self._cache = collections.defaultdict(list)

            # agent's state
            self._state = collections.defaultdict(lambda:
                self.rssm.get_initial_state(batch_size=1, dtype=self._dtype))
            self._prev_action = collections.defaultdict(lambda:
                tf.zeros((1, self._action_dim), self._dtype))

        def reset_states(self, worker_id, env_id):
            self._state[(worker_id, env_id)] = self._state.default_factory()
            self._prev_action[(worker_id, env_id)] = self._prev_action.default_factory()

        def __call__(self, worker_ids, env_ids, obs, evaluation=False):
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
            action, state = self.action(obs, state, prev_action, evaluation)

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

        def start(self, workers):
            objs = {workers[wid].reset_env.remote(eid): (wid, eid)
                for wid in range(self._n_workers) 
                for eid in range(self._envs_per_worker)}

            while True:
                ready_objs, not_objs = ray.wait(list(objs), self._action_batch)
                
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
                        self.finish_episode(wid, eid, o, r, d)
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
                    [self._cache[(wid, eid)].append(dict(prev_action=a))
                        for wid, eid, a in zip(wids, eids, actions)]

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
                    prev_action=np.zeros(self._action_shape, self._dtype),
                    reward=reward,
                    discount=discount
                ))

        def finish_episode(self, worker_id, env_id, obs, reward, discount):
            self.store_transition(worker_id, env_id, obs, reward, discount)
            episode = self._cache.pop((worker_id, env_id))
            episode = {k: convert_dtype([t[k] for t in episode], self._precision)
                for k in episode[0]}
            self.replay.merge(episode)
            score = np.sum(episode['reward'])
            epslen = len(episode['reward']) * self._frame_skip
            self.store(score=score, epslen=epslen)
            self.env_step += epslen

        def _learning(self):
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc('Learner starts learning...', color='blue')
            
            while True:
                start_train_step = self.train_step
                start_env_step = self.env_step
                start_time = time.time()
                self.learn_log(start_env_step)
                duration = time.time() - start_time
                self.store(
                    train_step=self.train_step,
                    env_step=self.env_step,
                    fps=(self.env_step - start_env_step)/duration,
                    tps=(self.train_step - start_train_step)/duration)
                self.log(self.env_step)
                self.save()

    return Learner


class Worker:
    def __init__(self, name, worker_id, env_config):
        cpu_affinity(f'Worker_{worker_id}')
        self.name = name
        self._id = worker_id
        self._n_envs = env_config['n_envs']
        env_config['n_workers'] = env_config['n_envs'] = 1
        self._envs = [create_env(env_config, make_env) 
            for _ in range(self._n_envs)]

    def reset_env(self, env_id):
        # return: obs, reward, discount, already_done
        return self._envs[env_id].reset(), 0, 1, False

    def env_step(self, env_id, action):
        obs, reward, done, _ = self._envs[env_id].step(action)
        discount = 1 - done
        already_done = self._envs[env_id].already_done()
        return obs, reward, discount, already_done

def get_worker_class():
    return Worker
