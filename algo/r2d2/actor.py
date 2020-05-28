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
from utility.run import Runner, evaluate
from utility import pkg
from env.gym_env import create_env
from core.dataset import process_with_env, DataFormat, RayDataset


def get_learner_class(BaseAgent):
    class Learner(BaseAgent):
        def __init__(self,
                    config, 
                    model_config,
                    env_config,
                    model_fn,
                    replay):
            cpu_affinity('Learner')
            silence_tf_logs()
            configure_threads(config['n_cpus'], config['n_cpus'])
            configure_gpu()
            configure_precision(config['precision'])

            env = create_env(env_config)
            
            self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                env=env)

            dtype = {16: tf.float16, 32: tf.float32}[config['precision']]
            obs_dtype = env.obs_dtype if len(env.obs_shape) == 3 else dtype
            action_dtype = tf.int32 if env.is_action_discrete else tf.float32
            batch_size = config['batch_size']
            sample_size = config['sample_size']
            is_per = ray.get(replay.name.remote())
            store_state = config['store_state']
            data_format = pkg.import_module('agent', config=config).get_data_format(
                env, batch_size, sample_size, is_per, store_state, self.models['q'].state_size
            )
            process = functools.partial(process_with_env, env=env, obs_range=[0, 1])
            dataset = RayDataset(replay, data_format, process)

            super().__init__(
                name='dq',
                config=config, 
                models=self.models,
                dataset=dataset,
                env=env,
            )

            self._log_locker = threading.Lock()
            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
            start_time = time.time()
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc(f'{self.name} starts learning...', color='blue')

            to_log = Every(self.LOG_PERIOD)
            while self.train_step < self.MAX_STEPS:
                start_train_step = self.train_step
                start_env_step = self.env_step
                start_time = time.time()
                self.learn_log(start_env_step)
                if to_log(self.train_step) and 'score' in self._logger and 'eval_score' in self._logger:
                    duration = time.time() - start_time
                    self.store(
                        train_step=self.train_step,
                        fps=(self.env_step - start_env_step) / duration,
                        tps=(self.train_step - start_train_step)/duration)
                    
                    with self._log_locker:
                        self.log(self.env_step)
                    self.save(print_terminal_info=False)

        def get_weights(self, name=None):
            return self.models.get_weights(name=name)

        def record_episode_info(self, **kwargs):
            with self._log_locker:
                self.store(**kwargs)
            if 'epslen' in kwargs:
                self.env_step += np.sum(kwargs['epslen'])

    return Learner


class Worker:
    @config
    def __init__(self, 
                *,
                worker_id,
                model_config,
                env_config, 
                buffer_config,
                model_fn,
                buffer_fn,):
        silence_tf_logs()
        configure_threads(1, 1)

        self._id = worker_id

        self.env = env = create_env(env_config)
        self.n_envs = self.env.n_envs
        self._action_dim = env.action_dim

        self.runner = Runner(self.env, self, nsteps=self.SYNC_PERIOD)

        self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                env=env)

        self._state = None

        self.buffer = buffer_fn(buffer_config, state_keys=['h', 'c'])
        self._is_per = buffer_config['type'].endswith('per')

        assert self.env.is_action_discrete == True
        self.q = self.models['q']
        self._pull_names = ['q']
        
        self._info = collections.defaultdict(list)
        if self._is_per:
            TensorSpecs = dict(
                obs=((self._sample_size, *env.obs_shape), env.obs_dtype, 'obs'),
                action=((self._sample_size,), tf.int32, 'action'),
                reward=((self._sample_size,), tf.float32, 'reward'),
                logpi=((self._sample_size,), tf.float32, 'logpi'),
                discount=((self._sample_size,), tf.float32, 'discount'),
                q=((self._sample_size,), tf.float32, 'q')
            )
            if self._store_state:
                state_size = self.q.state_size
                TensorSpecs['state'] = (
                    [((sz, ), tf.float32, name) 
                    for name, sz in zip(['h', 'c'], state_size)]
                )
            self.compute_priorities = build(self._compute_priorities, TensorSpecs)

    def __call__(self, obs, reset=np.zeros(1), deterministic=False):
        if self._state is None:
            self._state = self.q.get_initial_state(batch_size=self.n_envs)
        if np.any(reset):
            mask = tf.cast(1. - reset, tf.float32)
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
        prev_state = self._state
        action, terms, self._state = self.action(obs, self._state)
        if self._store_state:
            terms['h'] = prev_state[0]
            terms['c'] = prev_state[1]
        terms = tf.nest.map_structure(lambda x: np.squeeze(x.numpy()), terms)
        return action.numpy(), terms

    @tf.function
    def action(self, obs, state):
        qs, state = self.q.value(obs, state)
        qs = tf.squeeze(qs)
        q = tf.math.reduce_max(qs, axis=-1)
        action = tf.cast(tf.argmax(qs, axis=-1), tf.int32)
        rand_act = tfd.Categorical(tf.zeros_like(qs)).sample()
        eps_action = tf.where(
            tf.random.uniform(action.shape, 0, 1) < self._act_eps,
            rand_act, action)
        prob = tf.cast(eps_action == action, tf.float32)
        prob = prob * (1 - self._act_eps) + self._act_eps / self._action_dim
        logpi = tf.math.log(prob)
        return action, {'logpi': logpi, 'q': q}, state

    def run(self, learner, replay):
        while True:
            weights = self._pull_weights(learner)
            self._run(weights, replay)
            self._send_episode_info(learner)
        
    def _run(self, weights, replay):
        def collect(env, step, nth_obs, **kwargs):
            self.buffer.add(**kwargs)
            if self.buffer.is_full():
                self._send_data(replay)
            if env.already_done():
                self.buffer.reset()

        self.models.set_weights(weights)
        self.runner.run(step_fn=collect)
        
    def store(self, score, epslen):
        self._info['score'].append(score)
        self._info['epslen'].append(epslen)

    @tf.function
    def _compute_priorities(self, obs, action, reward, discount, logpi, state, q):
        embed = self.q.cnn(obs)
        if self._add_input:
            action_oh = tf.one_hot(action, self._action_dim, dtype=embed.dtype)
            rnn_input = tf.concat([embed, action_oh, reward[..., None]], -1)
        else:
            rnn_input = embed
        x, _ = self.q.rnn(rnn_input, state)
        next_x = x[:, 1:]
        next_qs = self.q.mlp(next_x)
        next_q = tf.math.reduce_max(next_qs, axis=-1)
        target_value = n_step_target(reward[:, :-1], next_q, discount[:, :-1], self._gamma, tbo=self._tbo)
        priority = tf.abs(target_value - q[:, :-1])
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([
            (next_q, (None, self._sample_size-1)),
            (target_value, (None, self._sample_size-1)),
            (q, (None, self._sample_size)),
            (priority, (None,))])

        return tf.squeeze(priority)

    def _pull_weights(self, learner):
        return ray.get(learner.get_weights.remote(name=self._pull_names))

    def _send_data(self, replay):
        data = self.buffer.sample()
        if self._is_per:
            data_tensor = {}
            for k, v in data.items():
                if 'obs' in k:
                    dtype = self.env.obs_dtype
                elif 'action' in k:
                    dtype = self.env.action_dtype
                else:
                    dtype = tf.float32
                v = np.expand_dims(v, 0)
                data_tensor[k] = tf.convert_to_tensor(v, dtype)
            del data['q']
            data['priority'] = self.compute_priorities(**data_tensor).numpy()
        replay.merge.remote(data, self.n_envs)

    def _send_episode_info(self, learner):
        if self._info:
            learner.record_episode_info.remote(**self._info)
            self._info.clear()

def get_worker_class():
    return Worker

class Evaluator:
    @config
    def __init__(self, 
                *,
                model_config,
                env_config,
                model_fn):
        silence_tf_logs()
        configure_threads(1, 1)

        self.env = env = create_env(env_config)
        self.n_envs = self.env.n_envs

        self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                env=env)

        self.q = self.models['q']
        self._pull_names = ['q']
        
        self._info = collections.defaultdict(list)

    def reset_states(self):
        self._state = self.q.get_initial_state(batch_size=self.n_envs)

    def __call__(self, x, deterministic=True, **kwargs):
        action, self._state = self.q.action(x, self._state, deterministic=True)
        return action.numpy()

    def run(self, learner):
        while True:
            weights = self._pull_weights(learner)
            self._run(weights)
            self._send_episode_info(learner)

    def _run(self, weights):
        self.models.set_weights(weights)
        score, epslen, _ = evaluate(self.env, self)
        self.store(score, epslen)

    def store(self, score, epslen):
        self._info['eval_score'].append(score)
        self._info['eval_epslen'].append(epslen)

    def _pull_weights(self, learner):
        return ray.get(learner.get_weights.remote(name=self._pull_names))

    def _send_episode_info(self, learner):
        if self._info:
            learner.record_episode_info.remote(**self._info)
            self._info.clear()

def get_evaluator_class():
    return Evaluator
