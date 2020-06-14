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
from algo.apex.actor import get_base_learner_class, BaseEvaluator

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
            configure_threads(config['n_cpus'], config['n_cpus'])
            configure_gpu()
            configure_precision(config['precision'])

            env = create_env(env_config)
            
            self.models = Ensemble(
                model_fn=model_fn, 
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
                env, batch_size, sample_size, is_per, store_state, self.models['q'].state_size
            )
            process = functools.partial(process_with_env, env=env, obs_range=[0, 1])
            dataset = RayDataset(replay, data_format, process)

            super().__init__(
                name=env.name,
                config=config, 
                models=self.models,
                dataset=dataset,
                env=env,
            )

            self._log_locker = threading.Lock()
            self._is_learning = True

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

        self._state = None
        self._prev_action = None
        self._prev_reward = None

        self.runner = Runner(self.env, self, nsteps=self.SYNC_PERIOD)

        self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                env=env)

        self.buffer = buffer_fn(buffer_config, state_keys=['h', 'c'])
        self._is_per = buffer_config['type'].endswith('per')

        assert self.env.is_action_discrete == True
        self.q = self.models['q']
        self._pull_names = ['q']
        
        self._info = collections.defaultdict(list)
        if self._is_per:
            TensorSpecs = dict(
                obs=((self._sample_size+1, *env.obs_shape), env.obs_dtype, 'obs'),
                prev_action=((self._sample_size+1,), tf.int32, 'prev_action'),
                prev_reward=((self._sample_size+1,), tf.float32, 'prev_reward'),
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

    def __call__(self, obs, reset=np.zeros(1), deterministic=False, env_output=None):
        if self._add_input:
            self._prev_reward = env_output.reward
        if self._state is None:
            self._state = self.q.get_initial_state(batch_size=self.n_envs)
            if self._add_input:
                self._prev_action = tf.zeros(self.n_envs)
                self._prev_reward = tf.zeros(self.n_envs)
        if np.any(reset):
            mask = tf.cast(1. - reset, tf.float32)
            self._state = tf.nest.map_structure(lambda x: x * mask, self._state)
            if self._add_input:
                self._prev_action = self._prev_action * mask
                self._prev_reward = self._prev_reward * mask
        action, terms, state = self.action(
                obs, self._state, self._prev_action, self._prev_reward)
        if self._store_state:
            terms['h'] = self._state[0]
            terms['c'] = self._state[1]
        terms = tf.nest.map_structure(lambda x: np.squeeze(x.numpy()), terms)
        self._state = state
        if self._add_input:
            self._prev_action = action
        return action.numpy(), terms

    @tf.function
    def action(self, obs, state, prev_action, prev_reward):
        qs, state = self.q.value(obs, state, prev_action, prev_reward)
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
        def reset_fn(obs, reward, **kwargs):
            self.buffer.pre_add(obs=obs, prev_action=0, prev_reward=reward)
        def collect(env, step, info, obs, action, reward, next_obs, **kwargs):
            kwargs['obs'] = next_obs
            kwargs['prev_action'] = action
            kwargs['prev_reward'] = reward
            self.buffer.add(**kwargs)
            if self.buffer.is_full():
                self._send_data(replay)
            if env.already_done():
                self.buffer.reset()

        self.models.set_weights(weights)
        self.runner.run(reset_fn=reset_fn, step_fn=collect)
        
    def store(self, score, epslen):
        self._info['score'].append(score)
        self._info['epslen'].append(epslen)

    @tf.function
    def _compute_priorities(self, obs, prev_action, prev_reward, discount, logpi, state, q):
        embed = self.q.cnn(obs)
        if self._add_input:
            pa, pr = prev_action, prev_reward
        else:
            pa, pr = None, None
        x, _ = self.q.rnn(embed, state,
            prev_action=pa, prev_reward=pr)
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
            (next_q, (None, self._sample_size)),
            (target_value, (None, self._sample_size)),
            (q, (None, self._sample_size)),
            (priority, (None,))])

        return tf.squeeze(priority)

    def _pull_weights(self, learner):
        return ray.get(learner.get_weights.remote(name=self._pull_names))

    def _send_data(self, replay):
        data = self.buffer.sample()
        if self._is_per:
            data_tensor = {k: tf.expand_dims(v, 0) for k, v in data.items()}
            del data['q']
            data['priority'] = self.compute_priorities(**data_tensor).numpy()
        replay.merge.remote(data)

    def _send_episode_info(self, learner):
        if self._info:
            learner.record_episode_info.remote(**self._info)
            self._info.clear()

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

        self.models = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                env=env)

        self.q = self.models['q']
        self._pull_names = ['q']
        
        self._state = None
        self._prev_action = None
        self._prev_reward = None

        self._info = collections.defaultdict(list)

    def reset_states(self):
        self._state = self.q.get_initial_state(batch_size=self.n_envs)
        if self._add_input:
            self._prev_action = tf.zeros(self.n_envs)
            self._prev_reward = tf.zeros(self.n_envs)

    def __call__(self, x, deterministic=True, env_output=None, **kwargs):
        if self._add_input:
            self._prev_reward = env_output.reward
        action, self._state = self.q.action(x, self._state, deterministic=True)
        if self._add_input:
            self._prev_action = action
        return action.numpy()

def get_evaluator_class():
    return Evaluator
