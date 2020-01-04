import numpy as np
import tensorflow as tf
import ray

from utility.timer import TBTimer
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
                models, 
                env,
                buffer,
                actor,
                value,
                config):        
        self.id = worker_id

        self.env = env
        self.n_envs = env.n_envs

        # models
        self.model = models
        self.actor = actor
        self.value = value

        self.buffer = buffer

        # args for priority replay
        self.per_alpha = config['per_alpha']
        self.per_epsilon = config['per_epsilon']
        
        self.log_steps = self.LOG_STEPS

        TensorSpecs = [
            (env.state_shape, tf.float32, 'state'),
            (env.action_shape, env.action_dtype, 'action'),
            ([1], tf.float32, 'reward'),
            (env.state_shape, tf.float32, 'next_state'),
            ([1], tf.float32, 'done'),
            ([1], tf.float32, 'steps')
        ]
        self.compute_priorities = tf_config.build(
            self._compute_priorities, 
            TensorSpecs)

    def eval_model(self, weights, step, replay):
        """ collects data, logs stats, and saves models """
        def collect_fn(**kwargs):
            self.buffer.add_data(**kwargs)

        self.model.set_weights(weights)

        with TBTimer(f'{self.name} -- eval model', self.TIME_PERIOD, to_log=self.timer):
            scores, epslens = run(self.env, self.actor, fn=collect_fn)
            step += np.sum(epslens)
            if scores is not None:
                self.store(  
                    score=np.mean(scores), 
                    score_std=np.std(scores),
                    score_max=np.max(scores), 
                    epslen=np.mean(epslens), 
                    epslen_std=np.std(epslens))
        
        return step, scores, epslens

    def pull_weights(self, learner):
        """ pulls weights from learner """
        with TBTimer(f'{self.name} pull weights', self.TIME_PERIOD, to_log=self.timer):
            return ray.get(learner.get_weights.remote(name=['actor', 'q1']))

    @tf.function
    def _compute_priorities(self, state, action, reward, next_state, done, steps):
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
            self.log(step=step, print_terminal_info=False)
            self.log_steps += self.LOG_STEPS
