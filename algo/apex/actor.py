import threading
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
import ray

from core.module import Ensemble
from core.tf_config import *
from core.base import BaseAgent
from core.decorator import agent_config
from utility.display import pwc
from utility.utils import Every
from utility.rl_utils import n_step_target
from utility.ray_setup import cpu_affinity
from utility.run import run
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
            self._dtype = global_policy().compute_dtype

            env = create_env(env_config)
            data_format = dict(
                obs=DataFormat((None, *env.obs_shape), self._dtype),
                action=DataFormat((None, *env.action_shape), self._dtype),
                reward=DataFormat((None, ), self._dtype), 
                nth_obs=DataFormat((None, *env.obs_shape), self._dtype),
                done=DataFormat((None, ), self._dtype),
            )
            if ray.get(replay.buffer_type.remote()).endswith('proportional'):
                data_format['IS_ratio'] = DataFormat((None, ), self._dtype)
                data_format['saved_idxes'] = DataFormat((None, ), tf.int32)
            if config['n_steps'] > 1:
                data_format['steps'] = DataFormat((None, ), self._dtype)
            if config['algorithm'].endswith('il'):
                data_format.update(dict(
                    mu=DataFormat((None, *env.action_shape), self._dtype),
                    std=DataFormat((None, *env.action_shape), self._dtype),
                    kl_flag=DataFormat((None, ), self._dtype),
                ))
            print(data_format)
            process = functools.partial(process_with_env, env=env)
            dataset = RayDataset(replay, data_format, process)

            self.models = Ensemble(
                model_fn=model_fn, 
                model_config=model_config, 
                action_dim=env.action_dim, 
                is_action_discrete=env.is_action_discrete)

            super().__init__(
                name=name, 
                config=config, 
                models=self.models,
                dataset=dataset,
                env=env,
            )
            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            pwc(f'{self.name} starts learning...', color='blue')
            step = 0
            self._writer.set_as_default()
            while True:
                step += 1
                self.learn_log(step)
                if step % 1000 == 0:
                    self.log(step, print_terminal_info=False)
                if step % 100000 == 0:
                    self.save(print_terminal_info=False)

        def get_weights(self, worker_id, name=None):
            return self.models.get_weights(name=name)

    return Learner


class BaseWorker(BaseAgent):
    """ This Base class defines some auxiliary functions for workers using PER
    """
    # currently, we have to define a separate base class in another file 
    # in order to utilize tf.function in ray.(ray version: 0.8.0dev6)
    @agent_config
    def __init__(self, 
                *,
                worker_id,
                env,
                buffer,
                actor,
                value):        
        self._id = worker_id

        self.env = env
        self.n_envs = env.n_envs

        # models
        self.models = Ensemble(models=self._ckpt_models)
        self.actor = actor
        self.value = value

        self.buffer = buffer

        self._to_log = Every(self.LOG_INTERVAL, start=self.LOG_INTERVAL)

        self._pull_names = ['actor'] if self._replay_type.endswith('uniform') else ['actor', 'q1']

        # args for priority replay
        if not self._replay_type.endswith('uniform'):
            TensorSpecs = [
                (env.obs_shape, env.obs_dtype, 'obs'),
                (env.action_shape, env.action_dtype, 'action'),
                ((), tf.float32, 'reward'),
                (env.obs_shape, env.obs_dtype, 'nth_obs'),
                ((), tf.float32, 'done'),
                ((), tf.float32, 'steps')
            ]
            self.compute_priorities = build(
                self._compute_priorities, 
                TensorSpecs)

    def set_weights(self, weights):
        self.models.set_weights(weights)

    def eval_model(self, weights, step, env=None, buffer=None, evaluation=False, tag='Learned', store_data=True):
        """ collects data, logs stats, and saves models """
        buffer = buffer or self.buffer
        def collect_fn(step, **kwargs):
            self._collect_data(buffer, store_data, tag, step, **kwargs)

        self.set_weights(weights)
        env = env or self.env
        scores, epslens = run(env, self.actor, fn=collect_fn, 
                                evaluation=evaluation, step=step)
        step += np.sum(epslens)
        
        return step, scores, epslens

    def pull_weights(self, learner):
        """ pulls weights from learner """
        return ray.get(learner.get_weights.remote(self._id, name=self._pull_names))

    def get_weights(self, name=None):
        return self.models.get_weights(name=name)

    def _collect_data(self, buffer, store_data, tag, step, **kwargs):
        if store_data:
            buffer.add_data(**kwargs)
        self._periodic_logging(step)

    def _send_data(self, replay, buffer=None, target_replay='fast_replay'):
        """ sends data to replay """
        buffer = buffer or self.buffer
        mask, data = buffer.sample()

        if not self._replay_type.endswith('uniform'):
            data_tensors = {k: tf.convert_to_tensor(v, tf.float32) for k, v in data.items()}
            data['priority'] = np.squeeze(self.compute_priorities(**data_tensors).numpy())

        replay.merge.remote(data, data['obs'].shape[0], target_replay=target_replay)

        buffer.reset()
        
    @tf.function
    def _compute_priorities(self, obs, action, reward, nth_obs, done, steps):
        if obs.dtype == tf.uint8:
            obs = tf.cast(obs, tf.float32) / 255.
            nth_obs = tf.cast(nth_obs, tf.float32) / 255.
        if self.env.is_action_discrete:
            action = tf.one_hot(action, self.env.action_dim)
        gamma = self.buffer.gamma
        value = self.value.step(obs, action)
        next_action, _ = self.actor._action(nth_obs, tf.convert_to_tensor(False))
        next_value = self.value.step(nth_obs, next_action)
        
        target_value = n_step_target(reward, done, next_value, gamma, steps)
        
        priority = tf.abs(target_value - value)
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([(priority, (None,))])

        return priority

    def _periodic_logging(self, step):
        if self._to_log(step):
            self.set_summary_step(self._to_log.step())
            self._logging(step=self._to_log.step())


    def _logging(self, step):
        self.store(**self.get_value('score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('epslen', mean=True, std=True, min=True, max=True))
        self.log(step=step, print_terminal_info=False)


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
        silence_tf_logs()
        configure_threads(1, 1)
        configure_gpu()

        env = create_env(env_config)
        
        buffer_config['seqlen'] = env.max_episode_steps
        buffer = buffer_fn(buffer_config)

        models = Ensemble(
            model_fn=model_fn, 
            model_config=model_config, 
            action_dim=env.action_dim, 
            is_action_discrete=env.is_action_discrete)

        super().__init__(
            name=name,
            worker_id=worker_id,
            models=models,
            env=env,
            buffer=buffer,
            actor=models['actor'],
            value=models['q1'],
            config=config)
        
    def run(self, learner, replay):
        step = 0
        log_time = self.LOG_INTERVAL
        while step < self.MAX_STEPS:
            weights = self.pull_weights(learner)

            step, scores, epslens = self.eval_model(weights, step)

            self._log_episodic_info(scores, epslens)

            self._send_data(replay)

            score = np.mean(scores)
            
            if step > log_time:
                self.save(print_terminal_info=False)
                log_time += self.LOG_INTERVAL

    def _log_episodic_info(self, scores, epslens):
        if scores is not None:
            self.store(
                score=scores,
                epslen=epslens,
            )

def get_worker_class():
    return Worker
