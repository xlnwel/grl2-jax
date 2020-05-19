import time
import threading
import functools
import collections
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import ray

from core.module import Ensemble
from core.tf_config import *
from core.base import BaseAgent
from core.decorator import config
from utility.display import pwc
from utility.utils import Every
from utility.rl_utils import n_step_target
from utility.ray_setup import cpu_affinity
from utility.run import Runner
from env.gym_env import create_env
from replay.data_pipline import process_with_env, DataFormat, RayDataset


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
            cpu_affinity('Learner')
            silence_tf_logs()
            configure_threads(config['n_cpus'], config['n_cpus'])
            configure_gpu()
            configure_precision(config['precision'])

            env = create_env(env_config)
            
            dtype = tf.float16 if config['precision'] == 16 else tf.float32
            action_dtype = tf.int32 if env.is_action_discrete else dtype
            data_format = dict(
                obs=DataFormat((None, *env.obs_shape), dtype),
                action=DataFormat((None, *env.action_shape), action_dtype),
                reward=DataFormat((None, ), dtype), 
                nth_obs=DataFormat((None, *env.obs_shape), dtype),
                discount=DataFormat((None, ), dtype),
            )
            if ray.get(replay.buffer_type.remote()).endswith('proportional'):
                data_format['IS_ratio'] = DataFormat((None, ), dtype)
                data_format['idxes'] = DataFormat((None, ), tf.int32)
            if config['n_steps'] > 1:
                data_format['steps'] = DataFormat((None, ), dtype)
            print(data_format)
            process = functools.partial(process_with_env, env=env)
            dataset = RayDataset(replay, data_format, process)

            self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                action_dim=env.action_dim, 
                is_action_discrete=env.is_action_discrete)

            super().__init__(
                name=name, 
                config=config, 
                models=self.models,
                dataset=dataset,
                env=env,
            )
            
            self._env_step = 0
            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            start_time = time.time()
            start_env_step = 0
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc(f'{self.name} starts learning...', color='blue')

            to_log = Every(self.LOG_INTERVAL)
            train_step = 0
            start_train_step = train_step
            while train_step < self.MAX_STEPS:
                self.learn_log(train_step)
                train_step += 1
                if to_log(train_step):
                    duration = time.time() - start_time
                    self.store(
                        train_step=train_step,
                        fps=(self._env_step - start_env_step) / duration,
                        tps=(train_step - start_train_step)/duration)
                    start_env_step = self._env_step
                    self.log(self._env_step)
                    self.save(print_terminal_info=False)
                    start_train_step = train_step
                    start_time = time.time()

        def get_weights(self, worker_id, name=None):
            return self.models.get_weights(name=name)

        def record_episode_info(self, score, epslen):
            self.store(score=score, epslen=epslen)
            self._env_step += np.sum(epslen)

    return Learner


class BaseWorker:
    @config
    def __init__(self, 
                *,
                name,
                worker_id,
                model_fn,
                buffer_fn,
                model_config,
                env_config, 
                buffer_config):
        silence_tf_logs()
        configure_threads(1, 1)

        self._id = worker_id

        self.env = env = create_env(env_config)
        self.n_envs = self.env.n_envs

        self.runner = Runner(self.env, self)

        self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                action_dim=env.action_dim, 
                is_action_discrete=env.is_action_discrete)

        if buffer_config['seqlen'] == 0:
            buffer_config['seqlen'] = env.max_episode_steps
        self.buffer = buffer_fn(buffer_config)
        self._is_per = buffer_config['type'].endswith('proportional')

        if 'actor' in self.models:
            self.actor = self.models['actor']
            self.value = self.models['q1']
            self._pull_names = ['actor', 'q1'] if self._is_per else ['actor']
        else:
            self.actor = self.value = self.models['q']
            self._pull_names = ['q']
        self._pull_names = ['actor', 'q1'] if self._is_per else ['actor']

        self._info = collections.defaultdict(list)

        if self._is_per:
            TensorSpecs = [
                (env.obs_shape, env.obs_dtype, 'obs'),
                (env.action_shape, env.action_dtype, 'action'),
                ((), tf.float32, 'reward'),
                (env.obs_shape, env.obs_dtype, 'nth_obs'),
                ((), tf.float32, 'discount'),
                ((), tf.float32, 'steps')
            ]
            self.compute_priorities = build(self._compute_priorities, TensorSpecs)

    def __call__(self, obs, deterministic=False, **kwargs):
        return self.actor(obs, deterministic, self._act_eps)
        
    def store(self, score, epslen):
        self._info['score'].append(score)
        self._info['epslen'].append(epslen)
        
    def get_weights(self, name=None):
        return self.models.get_weights(name=name)

    def _run(self, weights, env=None, buffer=None, evaluation=False, tag='Learned', store_data=True):
        """ collects data, logs stats, and saves models """
        buffer = buffer or self.buffer
        def collect_fn(env, step, **kwargs):
            self._collect_data(buffer, store_data, tag, step, **kwargs)

        self._set_weights(weights)
        self.runner.run_traj(step_fn=collect_fn)
        
    def _pull_weights(self, learner):
        """ pulls weights from learner """
        return ray.get(learner.get_weights.remote(self._id, name=self._pull_names))

    def _collect_data(self, buffer, store_data, tag, step, **kwargs):
        if store_data:
            buffer.add_data(**kwargs)

    def _set_weights(self, weights):
        self.models.set_weights(weights)

    def _send_data(self, replay, buffer=None, target_replay='fast_replay'):
        """ sends data to replay """
        buffer = buffer or self.buffer
        mask, data = buffer.sample()

        if self._is_per:
            data_tensors = {k: tf.convert_to_tensor(v, tf.float32) for k, v in data.items()}
            data['priority'] = np.squeeze(self.compute_priorities(**data_tensors).numpy())

        replay.merge.remote(data, data['obs'].shape[0], target_replay=target_replay)

        buffer.reset()

    @tf.function
    def _compute_priorities(self, obs, action, reward, nth_obs, discount, steps):
        if obs.dtype == np.uint8:
            obs = tf.cast(obs, tf.float32) / 255.
            nth_obs = tf.cast(nth_obs, tf.float32) / 255.
        if self.env.is_action_discrete:
            action = tf.one_hot(action, self.env.action_dim)
        gamma = self._gamma
        value = self.value.step(obs, action)
        nth_action, _ = self.action(nth_obs, False)
        nth_value = self.value.step(nth_obs, nth_action)
        
        target_value = n_step_target(reward, nth_value, gamma, discount, steps)
        
        priority = tf.abs(target_value - value)
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([(priority, (None,))])

        return priority


class Worker(BaseWorker):
    """ Interface """
    def __init__(self, 
                name,
                worker_id, 
                model_fn,
                buffer_fn,
                config,
                model_config, 
                env_config, 
                buffer_config):
        super().__init__(
            config=config,
            name=name,
            worker_id=worker_id,
            model_fn=model_fn,
            buffer_fn=buffer_fn,
            model_config=model_config,
            env_config=env_config,
            buffer_config=buffer_config)
        
    def run(self, learner, replay):
        while True:
            weights = self._pull_weights(learner)

            self._run(weights)

            self._send_data(replay)
            self._send_episode_info(learner)

    def _send_episode_info(self, learner):
        learner.record_episode_info.remote(**self._info)

def get_worker_class():
    return Worker
