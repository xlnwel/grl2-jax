import numpy as np
import tensorflow as tf
import ray

from utility.rl_utils import n_step_target
from core import tf_config
from core.decorator import agent_config
from core.base import BaseAgent
from env.gym_env import create_gym_env
from algo.run import run


class BaseWorker(BaseAgent):
    """ This Base class defines some auxiliary functions for workers using PER
    """
    # currently, we have to define a separate base class in another file 
    # in order to utilize tf.function in ray.(ray version: 0.8.0dev6)
    @agent_config
    def __init__(self, 
                *,
                name,
                worker_id,
                config,
                models, 
                env,
                buffer,
                actor,
                value):        
        self.id = worker_id

        self.env = env
        self.n_envs = env.n_envs

        # models
        self.model = models
        self.actor = actor
        self.value = value

        self.buffer = buffer

        self.log_steps = self.LOG_INTERVAL

        self.pull_names = ['actor'] if self.replay_type.endswith('uniform') else ['actor', 'q1']

        # args for priority replay
        if not self.replay_type.endswith('uniform'):
            self.per_alpha = config['per_alpha']
            self.per_epsilon = config['per_epsilon']

            TensorSpecs = [
                (env.state_shape, env.state_dtype, 'state'),
                (env.action_shape, env.action_dtype, 'action'),
                ([1], tf.float32, 'reward'),
                (env.state_shape, env.state_dtype, 'next_state'),
                ([1], tf.float32, 'done'),
                ([1], tf.float32, 'steps')
            ]
            self.compute_priorities = tf_config.build(
                self._compute_priorities, 
                TensorSpecs)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def eval_model(self, weights, step, env=None, buffer=None, evaluation=False, tag='Learned', store_exp=True):
        """ collects data, logs stats, and saves models """
        buffer = buffer or self.buffer
        def collect_fn(step, action_std, **kwargs):
            if store_exp:
                buffer.add_data(**kwargs)
            if np.any(action_std != 0):
                self.store(**{f'{tag}_action_std': np.mean(action_std)})
            self._periodic_logging(step)

        self.set_weights(weights)
        env = env or self.env
        scores, epslens = run(env, self.actor, fn=collect_fn, 
                                evaluation=evaluation, step=step, 
                                timer=self.timer, epsilon=self.act_eps)
        step += np.sum(epslens)
        
        return step, scores, epslens

    def pull_weights(self, learner):
        """ pulls weights from learner """
        return ray.get(learner.get_weights.remote(self.id, name=self.pull_names))

    def get_weights(self, name=None):
        return self.model.get_weights(name=name)

    def _send_data(self, replay, buffer=None, tag='Learned'):
        """ sends data to replay """
        buffer = buffer or self.buffer
        mask, data = buffer.sample()
            
        if not self.replay_type.endswith('uniform'):
            data_tensors = {k: tf.convert_to_tensor(v, tf.float32) for k, v in data.items()}
            for k in ['reward', 'done', 'steps']:
                data_tensors[k] = tf.expand_dims(data_tensors[k], -1)
            data['priority'] = np.squeeze(self.compute_priorities(**data_tensors).numpy())

        dest_replay = 'fast_replay' if tag == 'Learned' else 'slow_replay'
        replay.merge.remote(data, data['state'].shape[0], dest_replay=dest_replay)

        buffer.reset()
        
    @tf.function
    def _compute_priorities(self, state, action, reward, next_state, done, steps):
        if state.dtype == tf.uint8:
            state = tf.cast(state, tf.float32) / 255.
            next_state = tf.cast(next_state, tf.float32) / 255.
        if self.env.is_action_discrete:
            action = tf.one_hot(action, self.env.action_dim)
        gamma = self.buffer.gamma
        value = self.value.train_value(state, action)
        next_action = self.actor.train_action(next_state)
        next_value = self.value.train_value(next_state, next_action)
        
        target_value = n_step_target(reward, done, next_value, gamma, steps)
        
        priority = tf.abs(target_value - value)
        priority += self.per_epsilon
        priority **= self.per_alpha

        return priority

    def _periodic_logging(self, step):
        if step > self.log_steps:
            if self._log_condition():
                self.set_summary_step(self.log_steps)
                self._logging(step=self.log_steps)
            self.log_steps += self.LOG_INTERVAL

    def _log_condition(self):
        raise NotImplementedError

    def _logging(self, step):
        raise NotImplementedError
