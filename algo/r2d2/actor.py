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
from utility.timer import Timer
from utility.rl_utils import n_step_target
from utility.ray_setup import cpu_affinity
from utility.run import Runner, evaluate
from utility import pkg
from env.gym_env import create_env
from core.dataset import process_with_env, DataFormat, RayDataset
from algo.apex.actor import get_base_learner_class, BaseWorker, BaseEvaluator


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
            configure_precision(config['precision'])

            env = create_env(env_config)
            
            self.model = model_fn(
                config=model_config, 
                env=env)

            # TODO: figure out a way to automatic generate data_format from replay
            obs_dtype = env.obs_dtype
            action_dtype =  env.action_dtype
            batch_size = config['batch_size']
            sample_size = config['sample_size']
            is_per = ray.get(replay.name.remote()).endswith('per')
            store_state = config['store_state']
            data_format = pkg.import_module('agent', config=config).get_data_format(
                env, batch_size, sample_size, is_per, 
                store_state, self.model['q'].state_size
            )
            process = functools.partial(process_with_env, 
                env=env, obs_range=[0, 1], one_hot_action=False)
            dataset = RayDataset(replay, data_format, process)

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


class Worker(BaseWorker):
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

        self._state = None
        self._prev_action = None
        self._prev_reward = None

        self.runner = Runner(self.env, self, nsteps=self.SYNC_PERIOD)

        self.model = model_fn(
            config=model_config, 
            env=env)

        self.buffer = buffer_fn(buffer_config, state_keys=['h', 'c'])
        self._is_per = self._replay_type.endswith('per')

        assert self.env.is_action_discrete == True
        self.q = self.model['q']
        self._pull_names = ['q']
        
        self._info = collections.defaultdict(list)
        if self._is_per:
            TensorSpecs = dict(
                obs=((self._sample_size, *env.obs_shape), env.obs_dtype, 'obs'),
                prev_action=((self._sample_size,), tf.int32, 'prev_action'),
                prev_reward=((self._sample_size,), tf.float32, 'prev_reward'),
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
            self.compute_priorities = build(
                self._compute_priorities, TensorSpecs)

    def __call__(self, obs, reset, deterministic=False, env_output=None):
        self._prev_reward = env_output.reward
        if self._state is None:
            self._state = self.q.get_initial_state(batch_size=self.n_envs)
            self._prev_action = tf.squeeze(tf.zeros(self.n_envs, dtype=tf.int32))
            self._prev_reward = np.squeeze(np.zeros(self.n_envs))
        if np.any(reset):
            mask = tf.cast(1. - reset, tf.float32)
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
            self._prev_action = self._prev_action * tf.cast(mask, self._prev_action.dtype)
            self._prev_reward = self._prev_reward * mask
        assert self._prev_action.dtype == tf.int32, self._prev_action.dtype
        action, terms, state = self.model.action(
            obs, self._state, 
            deterministic=deterministic,
            epsilon=self._act_eps, 
            prev_action=self._prev_action, 
            prev_reward=self._prev_reward)
        terms = tf.nest.map_structure(lambda x: x.numpy(), terms)
        terms['prev_action'] = self._prev_action.numpy()
        terms['prev_reward'] = self._prev_reward
        if self._store_state:
            terms['h'] = np.squeeze(self._state[0].numpy())
            terms['c'] = np.squeeze(self._state[1].numpy())
        self._state = state
        self._prev_action = action
        return action.numpy(), terms
        
    def _run(self, weights, replay):
        def collect(env, step, reset, action, reward, next_obs, **kwargs):
            self.buffer.add(**kwargs)
            if self.buffer.is_full():
                self._send_data(replay)
            if kwargs['discount'] == 0:
                self.buffer.reset()

        self.model.set_weights(weights)
        self.runner.run(step_fn=collect)

    @tf.function
    def _compute_priorities(self, obs, prev_action, prev_reward, discount, logpi, state, q):
        q = q[:, :-1]
        discount = discount[:, :-1]

        embed = self.q.cnn(obs)
        x, _ = self.q.rnn(embed, state,
            prev_action=prev_action, prev_reward=prev_reward)
        next_qs = self.q.mlp(x[:, 1:])
        # intend not to use the target net for computational efficiency
        # the effect is not tested, but I conjecture it won't be much
        next_q = tf.math.reduce_max(next_qs, axis=-1)
        target_value = n_step_target(prev_reward[:, 1:], next_q, discount, self._gamma, tbo=self._tbo)
        priority = tf.abs(target_value - q)
        priority = (self._per_eta*tf.math.reduce_max(priority, axis=1) 
                    + (1-self._per_eta)*tf.math.reduce_mean(priority, axis=1))
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([
            (next_q, (None, self._sample_size-1)),
            (target_value, (None, self._sample_size-1)),
            (q, (None, self._sample_size-1)),
            (priority, (None,))])

        return tf.squeeze(priority)

    def _send_data(self, replay):
        data = self.buffer.sample()
        if self._is_per:
            data_tensor = {k: tf.expand_dims(v, 0) for k, v in data.items()}
            del data['q']
            data['priority'] = self.compute_priorities(**data_tensor).numpy()
        replay.merge.remote(data)

def get_worker_class():
    return Worker

class Evaluator(BaseEvaluator):
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

        self.model = model_fn(
            config=model_config, 
            env=env)

        self.q = self.model['q']
        self._pull_names = ['q']
        
        self.reset_states()

        self._info = collections.defaultdict(list)

    def reset_states(self):
        self._state = self.q.get_initial_state(batch_size=self.n_envs)
        self._prev_action = tf.zeros(self.n_envs)
        self._prev_reward = tf.zeros(self.n_envs)

    def __call__(self, x, deterministic=True, env_output=None, **kwargs):
        self._prev_reward = env_output.reward
        action, _, self._state = self.model.action(
            x, self._state, deterministic, self._eval_act_eps,
            prev_action=self._prev_action,
            prev_reward=self._prev_reward)
        self._prev_action = action
        return action.numpy()

def get_evaluator_class():
    return Evaluator
