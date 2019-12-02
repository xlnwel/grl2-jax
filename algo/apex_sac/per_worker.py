import numpy as np
import tensorflow as tf
import ray

from utility.timer import Timer
from core import tf_config
from core.decorator import agent_config
from core.base import BaseAgent
from env.gym_env import create_gym_env
from algo.run import run_trajectory, run_trajectories


class PERWorker(BaseAgent):
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
                target_value,
                config):        
        self.id = worker_id

        self.env = env

        # models
        self.model = models
        self.ckpt_models.update(self.model)
        self.actor = actor
        self.value = value
        self.target_value = target_value

        self.buffer = buffer

        # args for priority replay
        self.per_alpha = config['per_alpha']
        self.per_epsilon = config['per_epsilon']
        
        TensorSpecs = [
            (env.state_shape, env.state_dtype, 'state'),
            ([env.action_dim], env.action_dtype, 'action'),
            ([1], tf.float32, 'reward'),
            (env.state_shape, tf.float32, 'next_state'),
            ([1], tf.float32, 'done'),
            ([1], tf.float32, 'steps')
        ]
        self.compute_priorities = tf_config.build(
            self._compute_priorities, 
            TensorSpecs)

    def eval_model(self, weights, episode_i, step):
        """ collects data, logs stats, and saves models """
        def collect_fn(state, action, reward, done, next_state=None, mask=None):
            mask = np.squeeze(mask)
            self.buffer.add_data(state, action, reward, done, next_state, mask)
        run_fn = run_trajectory if self.env.n_envs == 1 else run_trajectories

        episode_i += 1
        self.model.set_weights(weights)

        with Timer(f'Worker {self.id} -- eval', 10):
            scores, epslens = run_fn(self.env, self.actor, fn=collect_fn)
            step += np.sum(epslens)
            self.store(
                score= np.mean(scores), 
                score_std=np.std(scores),
                score_max=np.max(scores), 
                epslen=np.mean(epslens), 
                epslen_std=np.std(epslens))

        if episode_i % 10 == 0:
            # record stats
            self.log(step=episode_i, print_terminal_info=False)
            self.save(episode_i, print_terminal_info=False)
        
        return episode_i, step, scores
                    
    def send_data(self, learner, env_mask=None):
        """ sends data to learner """
        data = self.buffer.sample(env_mask)
        data_tesnors = {k: tf.convert_to_tensor(v, tf.float32) for k, v in data.items()}
        data['priority'] = self.compute_priorities(**data_tesnors).numpy()
        learner.merge_buffer.remote(data, data['state'].shape[0])
        self.buffer.reset()

    def pull_weights(self, learner):
        """ pulls weights from learner """
        return ray.get(learner.get_weights.remote())

    @tf.function
    def _compute_priorities(self, state, action, reward, next_state, done, steps):
        gamma = self.buffer.gamma
        value = self.value.train_step(state, action)
        next_action, next_logpi = self.actor.train_step(next_state)
        next_value = self.target_value.train_step(next_state, next_action)
        # for brevity, we don't compute n-th value and use n-th q directly here
        target_value = reward + gamma**steps * (1-done) * next_value
        
        priority = tf.abs(target_value - value)
        priority += self.per_epsilon
        priority **= self.per_alpha

        return priority
