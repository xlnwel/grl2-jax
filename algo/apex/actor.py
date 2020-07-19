import time
import threading
import functools
import collections
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import ray

from core.tf_config import *
from core.base import BaseAgent
from core.decorator import config
from utility.display import pwc
from utility.utils import Every
from utility.graph import video_summary
from utility.timer import Timer
from utility.rl_utils import n_step_target
from utility.ray_setup import cpu_affinity
from utility.run import Runner, evaluate
from utility import pkg
from env.gym_env import create_env
from core.dataset import process_with_env, DataFormat, RayDataset


pull_names = dict(
    dqn=['q'],
    iqn=['q'],
    iqncrl=['q'],
    fqf=['q', 'fpn'],
    sac=['actor', 'q1'],
    sacd=['cnn', 'actor', 'q1'],
    sacdiqn=['cnn', 'actor', 'q1'],
    sacdiqn2=['cnn', 'actor', 'q'],
)

def get_pull_names(algo):
    algo = algo.rsplit('-', 1)[-1]
    if algo not in pull_names and algo[-1].isdigit():
        algo = algo[:-1]
    return pull_names[algo]
    

def get_base_learner_class(BaseAgent):
    class BaseLearner(BaseAgent):            
        def is_learning(self):
            return self._is_learning

        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            start_time = time.time()
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc(f'{self.name} starts learning...', color='blue')

            to_log = Every(self.LOG_PERIOD)
            while self.env_step < self.MAX_STEPS:
                start_train_step = self.train_step
                start_env_step = self.env_step
                start_time = time.time()
                self.learn_log(start_env_step)
                if to_log(self.train_step) and 'eval_score' in self._logger:
                    duration = time.time() - start_time
                    self.store(
                        train_step=self.train_step,
                        fps=(self.env_step - start_env_step) / duration,
                        tps=(self.train_step - start_train_step)/duration)
                    with self._log_locker:
                        self.log(self.env_step)
                    self.save(print_terminal_info=False)
            
            self._is_learning = False

        def get_weights(self, name=None):
            return self.model.get_weights(name=name)

        def record_episode_info(self, worker_id=None, **kwargs):
            video = kwargs.pop('video', None)
            if 'epslen' in kwargs:
                self.env_step += np.sum(kwargs['epslen'])
            with self._log_locker:
                if self._schedule_act_eps and worker_id is not None:
                    kwargs = {f'{k}_{worker_id}': v for k, v in kwargs.items()}
                self.store(**kwargs)
            if video is not None:
                video_summary(f'{self.name}/sim', video, step=self.env_step)

    return BaseLearner

def get_learner_class(BaseAgent):
    BaseLearner = get_base_learner_class(BaseAgent)
    class Learner(BaseLearner):
        def __init__(self,
                    config, 
                    model_config,
                    env_config,
                    model_fn,
                    replay):
            cpu_affinity('Learner')
            silence_tf_logs()
            configure_threads(config['n_learner_cpus'], config['n_learner_cpus'])
            configure_gpu()
            configure_precision(config.get('precision', 32))

            env = create_env(env_config)
            
            obs_dtype = env.obs_dtype
            action_dtype =  env.action_dtype
            algo = config['algorithm'].split('-', 1)[-1]
            is_per = ray.get(replay.name.remote()).endswith('per')
            n_steps = config['n_steps']
            data_format = pkg.import_module('agent', algo).get_data_format(
                env, is_per, n_steps)
            process = functools.partial(process_with_env, env=env)
            dataset = RayDataset(replay, data_format, process)

            self.model = model_fn(
                config=model_config, 
                env=env)

            super().__init__(
                name=env.name,
                config=config, 
                models=self.model,
                dataset=dataset,
                env=env,
            )

            self._log_locker = threading.Lock()
            self._is_learning = True
            
    return Learner


class BaseWorker:
    def run(self, learner, replay):
        while True:
            weights = self._pull_weights(learner)
            self._run(weights, replay)
            self._send_episode_info(learner)

    def store(self, score, epslen):
        self._info['score'].append(score)
        self._info['epslen'].append(epslen)

    def _pull_weights(self, learner):
        return ray.get(learner.get_weights.remote(name=self._pull_names))

    def _send_episode_info(self, learner):
        if self._info:
            learner.record_episode_info.remote(self._id, **self._info)
            self._info.clear()

class Worker(BaseWorker):
    @config
    def __init__(self, 
                *,
                worker_id,
                model_config,
                env_config, 
                buffer_config,
                model_fn,
                buffer_fn):
        silence_tf_logs()
        configure_threads(1, 1)
        self._id = worker_id

        self.env = env = create_env(env_config)
        self.n_envs = self.env.n_envs

        if buffer_config['seqlen'] == 0:
            buffer_config['seqlen'] = env.max_episode_steps // getattr(env, 'frame_skip', 1)
        self._seqlen = buffer_config['seqlen']
        self.buffer = buffer_fn(buffer_config)
        self._is_per = self._replay_type.endswith('per')

        self.runner = Runner(self.env, self, nsteps=self.SYNC_PERIOD)

        self.model = model_fn( 
            config=model_config, 
            env=env)

        self._is_dpg = 'actor' in self.model
        self._is_iqn = 'iqn' in self._algorithm or 'fqf' in self._algorithm
        for k, v in self.model.items():
            setattr(self, k, v)
        
        self._pull_names = get_pull_names(self._algorithm)
        
        self._info = collections.defaultdict(list)

        if self._is_per:
            TensorSpecs = dict(
                obs=(env.obs_shape, env.obs_dtype, 'obs'),
                action=(env.action_shape, env.action_dtype, 'action'),
                reward=((), tf.float32, 'reward'),
                next_obs=(env.obs_shape, env.obs_dtype, 'next_obs'),
                discount=((), tf.float32, 'discount'),
                steps=((), tf.float32, 'steps')
            )
            if not self._is_dpg:
                if self._is_iqn:
                    TensorSpecs['qtv'] = ((self.K,), tf.float32, 'qtv')
                else:
                    TensorSpecs['q']= ((), tf.float32, 'q')
            if self._is_iqn:
                self.compute_priorities = build(
                    self._compute_iqn_priorities, TensorSpecs, batch_size=self._seqlen)
            else:
                self.compute_priorities = build(
                    self._compute_dqn_priorities, TensorSpecs)

    def __call__(self, x, deterministic=False, **kwargs):
        action = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=deterministic,
            epsilon=self._act_eps)
        action = tf.nest.map_structure(lambda x: x.numpy(), action)
        return action

    def _run(self, weights, replay):
        def collect(env, step, reset, **kwargs):
            if reset:
                kwargs['next_obs'] = env.prev_obs()
            self.buffer.add_data(**kwargs)
            if self.buffer.is_full():
                self._send_data(replay)

        self.model.set_weights(weights)
        if self._seqlen == 0:
            self.runner.run_traj(step_fn=collect)
        else:
            self.runner.run(step_fn=collect)

    def _send_data(self, replay, buffer=None):
        buffer = buffer or self.buffer
        data = buffer.sample()

        if self._is_per:
            data_tensor = {k: tf.convert_to_tensor(v) for k, v in data.items()}
            data['priority'] = self.compute_priorities(**data_tensor).numpy()
        if self._is_iqn:
            data.pop('qtv', None)
        else:
            data.pop('q', None)
        replay.merge.remote(data, data['action'].shape[0])
        buffer.reset()

    @tf.function
    def _compute_dqn_priorities(self, obs, action, reward, next_obs, discount, steps, q=None):
        if self._is_dpg:
            q = self.q(obs, action)
            next_action = self.actor(next_obs, deterministic=False)
            next_q = self.q(next_obs, next_action)
        else:
            next_action = self.q.action(next_obs, False)
            next_q = self.q.value(next_obs, next_action)
            
        returns = n_step_target(reward, next_q, discount, self._gamma, steps, self._tbo)
        
        priority = tf.abs(returns - q)
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([(priority, (None,))])

        return tf.squeeze(priority)
    
    @tf.function
    def _compute_iqn_priorities(self, obs, action, reward, next_obs, discount, steps, qtv=None):
        next_action = self.q.action(next_obs, self.K)
        _, next_qtv, _ = self.q.value(next_obs, self.N_PRIME, next_action)
        reward = reward[:, None, None]
        discount = discount[:, None, None]
        steps = steps[:, None, None]
        returns = n_step_target(reward, next_qtv, discount, self._gamma, steps, self._tbo)
        returns = tf.transpose(returns, (0, 2, 1))      # [B, 1, N']
        qtv = tf.expand_dims(qtv, axis=-1)              # [B, K, 1], to avoid redundant computation, we use previously computed qtv here
        tf.debugging.assert_shapes([[qtv, (self._seqlen, self.K, 1)]])
        tf.debugging.assert_shapes([[returns, (self._seqlen, 1, self.N_PRIME)]])
        
        error = tf.abs(returns - qtv)
        priority = tf.reduce_max(tf.reduce_mean(error, axis=2), axis=1)
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([(priority, (None,))])

        return tf.squeeze(priority)

def get_worker_class():
    return Worker


class BaseEvaluator:
    def run(self, learner):
        step = 0
        if getattr(self, 'RECORD_PERIOD', False):
            to_record = Every(self.RECORD_PERIOD)
        else:
            to_record = lambda x: False 
        while True:
            step += 1
            weights = self._pull_weights(learner)
            self._run(weights, record=to_record(step))
            self._send_episode_info(learner)

    def _run(self, weights, record):        
        self.model.set_weights(weights)
        score, epslen, video = evaluate(self.env, self, 
            record=record, n=self.N_EVALUATION)
        self.store(score, epslen, video)

    def store(self, score, epslen, video):
        self._info['eval_score'] += score
        self._info['eval_epslen'] += epslen
        if video is not None:
            self._info['video'] = video

    def _pull_weights(self, learner):
        return ray.get(learner.get_weights.remote(name=self._pull_names))

    def _send_episode_info(self, learner):
        if self._info:
            learner.record_episode_info.remote(**self._info)
            self._info.clear()


class Evaluator(BaseEvaluator):
    @config
    def __init__(self, 
                *,
                model_config,
                env_config,
                model_fn):
        silence_tf_logs()
        configure_threads(1, 1)

        env_config.pop('reward_clip', False)
        self.env = env = create_env(env_config)
        self.n_envs = self.env.n_envs

        self.model = model_fn(
                config=model_config, 
                env=env)

        self._pull_names = get_pull_names(self._algorithm)
        
        self._info = collections.defaultdict(list)

    def __call__(self, x, deterministic=True, **kwargs):
        action, terms = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=deterministic,
            epsilon=self._eval_act_eps)
        action = action.numpy()
        
        return action


def get_evaluator_class():
    return Evaluator
