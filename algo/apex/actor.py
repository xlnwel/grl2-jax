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
from core.decorator import record, config
from utility.display import pwc
from utility.utils import Every
from utility.timer import Timer
from utility.rl_utils import n_step_target
from utility.ray_setup import cpu_affinity
from utility.run import Runner, evaluate, RunMode
from utility import pkg
from env.func import create_env
from core.dataset import process_with_env, DataFormat, RayDataset


pull_names = dict(
    dqn=['q'],
    iqn=['q'],
    iqncrl=['q'],
    fqf=['q', 'fpn'],
    sac=['actor', 'q'],
    sacd=['encoder', 'actor', 'q'],
    sacdiqn=['encoder', 'actor', 'q'],
    sacdiqn3=['encoder', 'actor', 'v'],
    sacdiqn4=['encoder', 'actor', 'v'],
    sacdiqn5=['actor_encoder', 'critic_encoder', 'actor', 'v'],
    sacdiqncrl=['encoder', 'actor', 'q'],
    sacdiqncrlar=['encoder', 'actor', 'q'],
    sacdiqnmdp=['encoder', 'actor'],
    sacdiqnbs=['encoder', 'state', 'actor'],
)

def get_pull_names(algo):
    algo = algo.rsplit('-', 1)[-1]
    if algo not in pull_names and algo[-1].isdigit():
        algo = algo[:-1]
    names = pull_names[algo]
    print('pull names:', names)
    return names
    

def get_base_learner_class(BaseAgent):
    class BaseLearner(BaseAgent):            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            start_time = time.time()
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc(f'{self.name} starts learning...', color='blue')

            while True:
                self.learn_log()
            
        def get_weights(self, name=None):
            return self.model.get_weights(name=name)

        def get_stats(self):
            return self.train_step, super().get_stats()

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
            one_hot_action = config.get('one_hot_action', True)
            process = functools.partial(process_with_env, 
                env=env, one_hot_action=one_hot_action)
            dataset = RayDataset(replay, data_format, process)

            self.model = model_fn(
                config=model_config, 
                env=env)

            super().__init__(
                config=config, 
                models=self.model,
                dataset=dataset,
                env=env,
            )
            
    return Learner


class BaseWorker:
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
        configure_gpu()
        self._id = worker_id

        self.env = create_env(env_config)
        self.n_envs = self.env.n_envs

        self.model = model_fn( 
            config=model_config, 
            env=self.env)

        self._run_mode = getattr(self, '_run_mode', RunMode.NSTEPS)
        self.runner = Runner(
            self.env, self, 
            nsteps=self.SYNC_PERIOD if self._run_mode == RunMode.NSTEPS else None,
            run_mode=self._run_mode)
        
        assert self._run_mode in [RunMode.NSTEPS, RunMode.TRAJ]

    def run(self, learner, replay, monitor):
        while True:
            weights = self._pull_weights(learner)
            self._run(weights, replay)
            self._send_episode_info(monitor)

    def store(self, score, epslen):
        if isinstance(score, (int, float)):
            self._info['score'].append(score)
            self._info['epslen'].append(epslen)
        else:
            self._info['score'] += list(score)
            self._info['epslen'] += list(epslen)

    def _pull_weights(self, learner):
        return ray.get(learner.get_weights.remote(name=self._pull_names))

    def _send_episode_info(self, learner):
        if self._info:
            learner.record_episode_info.remote(self._id, **self._info)
            self._info.clear()

class Worker(BaseWorker):
    def __init__(self, 
                *,
                worker_id,
                config,
                model_config,
                env_config, 
                buffer_config,
                model_fn,
                buffer_fn):
        super().__init__(
            worker_id=worker_id,
            config=config,
            model_config=model_config,
            env_config=env_config,
            buffer_config=buffer_config,
            model_fn=model_fn,
            buffer_fn=buffer_fn
        )

        self._seqlen = buffer_config['seqlen']
        self.buffer = buffer_fn(buffer_config)

        self._return_stats = 'encoder' in self.model or 'actor' not in self.model
        self._is_iqn = 'iqn' in self._algorithm or 'fqf' in self._algorithm
        for k, v in self.model.items():
            setattr(self, k, v)
        
        self._pull_names = get_pull_names(self._algorithm)
        
        self._info = collections.defaultdict(list)
        if self._worker_side_prioritization:
            TensorSpecs = dict(
                obs=(self.env.obs_shape, self.env.obs_dtype, 'obs'),
                action=(self.env.action_shape, self.env.action_dtype, 'action'),
                reward=((), tf.float32, 'reward'),
                next_obs=(self.env.obs_shape, self.env.obs_dtype, 'next_obs'),
                discount=((), tf.float32, 'discount'),
                steps=((), tf.float32, 'steps')
            )
            if self._is_iqn:
                TensorSpecs['qtv'] = ((self.K,), tf.float32, 'qtv')
            else:
                TensorSpecs['q']= ((), tf.float32, 'q')
            
            self.compute_priorities = build(
                self._compute_iqn_priorities if self._is_iqn else self._compute_dqn_priorities, TensorSpecs)
        self._return_stats = self._worker_side_prioritization or buffer_config.get('max_steps', 0) > buffer_config.get('n_steps')
        print(f'{worker_id} action epsilon:', self._act_eps)
        print(f'{worker_id} action inv_temp:', np.squeeze(self.model.actor.act_inv_temp))

    def __call__(self, x, **kwargs):
        action = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=False,
            epsilon=tf.convert_to_tensor(self._act_eps, tf.float32),
            return_stats=self._return_stats)
        action = tf.nest.map_structure(lambda x: x.numpy(), action)
        return action

    def _run(self, weights, replay):
        def collect(env, step, reset, **kwargs):
            self.buffer.add_data(**kwargs)
            if self.buffer.is_full():
                self._send_data(replay)
        start_step = self.runner.step
        self.model.set_weights(weights)
        end_step = self.runner.run(step_fn=collect)
        return end_step - start_step

    def _send_data(self, replay, buffer=None):
        buffer = buffer or self.buffer
        data = buffer.sample()

        if self._worker_side_prioritization:
            data_tensor = {k: tf.convert_to_tensor(v) for k, v in data.items()}
            data['priority'] = self.compute_priorities(**data_tensor).numpy()
        data.pop('qtv', None)
        data.pop('q', None)
        replay.merge.remote(data, data['action'].shape[0])
        buffer.reset()

    @tf.function
    def _compute_dqn_priorities(self, obs, action, reward, next_obs, discount, steps, q=None):
        if self._return_stats:
            next_x = self.encoder(next_obs) if hasattr(self, 'encoder') else next_obs
            next_action = self.q.action(next_x, False)
            next_q = self.q.value(next_x, next_action)
        else:
            x = self.encoder(obs) if hasattr(self, 'encoder') else obs
            q = self.q(x, action)
            next_x = self.encoder(next_obs) if hasattr(self, 'encoder') else next_obs
            # sac results in probs while others sample actions
            next_act_probs, next_act_logps = self.actor.train_step(next_x)
            # we do not use the target model to save some bandwidth
            next_qs = self.q(next_x)
            _, temp = self.temperature()
            next_q = tf.reduce_sum(next_act_probs
                * (next_qs - temp * next_act_logps), axis=-1)
            
        returns = n_step_target(reward, next_q, discount, self._gamma, steps)
        
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
    def run(self, learner, monitor):
        step = 0
        if getattr(self, 'RECORD_PERIOD', False):
            to_record = Every(self.RECORD_PERIOD)
        else:
            to_record = lambda x: False 
        while True:
            step += 1
            weights = self._pull_weights(learner)
            self._run(weights, record=to_record(step))
            self._send_episode_info(monitor)

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
        if not hasattr(self, '_deterministic_evaluation'):
            self._deterministic_evaluation = True
        
        self._info = collections.defaultdict(list)

    def __call__(self, x, **kwargs):
        action = self.model.action(
            tf.convert_to_tensor(x), 
            deterministic=self._deterministic_evaluation,
            epsilon=self._eval_act_eps)
        if isinstance(action, tuple):
            if len(action) == 2:
                action, terms = action
                return action.numpy()
            elif len(action) == 3:
                action, ar, terms = action
                return action.numpy(), ar.numpy()
            else:
                raise ValueError(action)
        action = action.numpy()
        
        return action


def get_evaluator_class():
    return Evaluator
