import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.decorator import agent_config
from core.base import BaseAgent
from core.module import Ensemble
from utility.rl_utils import n_step_target
from utility.utils import Every
from env.gym_env import create_env
from algo.run import run


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

        self._should_log = Every(self.LOG_INTERVAL)

        self._pull_names = ['actor'] if self._replay_type.endswith('uniform') else ['actor', 'q1']

        # args for priority replay
        if not self._replay_type.endswith('uniform'):
            TensorSpecs = [
                (env.obs_shape, env.obs_dtype, 'obs'),
                (env.action_shape, env.action_dtype, 'action'),
                ((), tf.float32, 'reward'),
                (env.obs_shape, env.obs_dtype, 'next_obs'),
                ((), tf.float32, 'done'),
                ((), tf.float32, 'steps')
            ]
            self.compute_priorities = tf_config.build(
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
                                evaluation=evaluation, step=step, 
                                timer=self._timer, epsilon=self._act_eps)
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
    def _compute_priorities(self, obs, action, reward, next_obs, done, steps):
        if obs.dtype == tf.uint8:
            obs = tf.cast(obs, tf.float32) / 255.
            next_obs = tf.cast(next_obs, tf.float32) / 255.
        if self.env.is_action_discrete:
            action = tf.one_hot(action, self.env.action_dim)
        gamma = self.buffer.gamma
        value = self.value.step(obs, action)
        next_action, _ = self.actor._action(next_obs, tf.convert_to_tensor(False))
        next_value = self.value.step(next_obs, next_action)
        
        target_value = n_step_target(reward, done, next_value, gamma, steps)
        
        priority = tf.abs(target_value - value)
        priority += self._per_epsilon
        priority **= self._per_alpha

        tf.debugging.assert_shapes([(priority, (None,))])

        return priority

    def _periodic_logging(self, step):
        if self._log_condition() and self._should_log(step):
            self.set_summary_step(self._should_log.step())
            self._logging(step=self._should_log.step())

    def _log_condition(self):
        return True

    def _logging(self, step):
        self.store(**self.get_value('score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('epslen', mean=True, std=True, min=True, max=True))
        self.log(step=step, print_terminal_info=False)

