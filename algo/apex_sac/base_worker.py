import numpy as np
import tensorflow as tf
import ray

from core import log
from core import tf_config
from utility.timer import Timer
from env.gym_env import create_gym_env
from algo.run import run_trajectory, run_trajectories


class BaseWorker:
    # currently, we have to define a separate base class in another file 
    # in order to utilize tf.function in ray.(ray version: 0.8.0dev6)
    def __init__(self, 
                name,
                worker_id,
                model_fn, 
                buffer_fn,
                config,
                model_config, 
                env_config, 
                buffer_config, 
                to_log):
        tf_config.configure_threads(1, 1)
        tf_config.configure_gpu()
        
        self.id = worker_id                             # use 0 worker to evaluate the model

        self.env = env = create_gym_env(env_config)

        # models
        self.model = model_fn(model_config, env.state_shape, env.action_dim, env.is_action_discrete)
        self.actor = self.model['actor']
        self.softq = self.model['softq1']
        self.target_softq = self.model['target_softq1']

        buffer_config['epslen'] = env.max_episode_steps
        self.buffer = buffer_fn(
            buffer_config, env.state_shape, 
            env.state_dtype, env.action_dim, 
            env.action_dtype, config['gamma'])

        # args for priority replay
        self.per_alpha = config['per_alpha']
        self.per_epsilon = config['per_epsilon']
        
        self.to_log = to_log
        if to_log:
            self.writer = log.setup_tensorboard(config['log_root_dir'], config['model_name'])

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
        """ collects data and does some logging """
        def collect_fn(state, action, reward, done, next_state=None, mask=None):
            mask = np.squeeze(mask)
            self.buffer.add_data(state, action, reward, done, next_state, mask)
        run_fn = run_trajectory if self.env.n_envs == 1 else run_trajectories

        episode_i += 1
        self.model.set_weights(weights)

        with Timer(f'Worker {self.id} -- sample', 1000):
            scores, epslens = run_fn(self.env, self.actor, fn=collect_fn)
            step += np.sum(epslens)

        score_mean = np.mean(scores)
        if self.to_log and episode_i % 10 == 0:
            # record stats
            stats = dict(
                score=score_mean, 
                score_std=np.std(scores),
                score_max=np.max(scores), 
                epslen=np.mean(epslens), 
                epslen_std=np.std(epslens), 
            )
            log.log_summary(self.writer, stats, step=episode_i)
        
        return episode_i, step, score_mean
                    
    def send_data(self, learner):
        """ sends data to learner """
        data = self.buffer.sample()
        data_tesnors = {k: tf.convert_to_tensor(v, tf.float32) for k, v in data.items()}
        data['priority'] = self.compute_priorities(**data_tesnors).numpy()
        learner.merge_buffer.remote(data, data['state'].shape[0])
        self.buffer.reset()

    def _pull_weights(self, learner):
        """ pulls weights from learner """
        return ray.get(learner.get_weights.remote())

    @tf.function
    def _compute_priorities(self, state, action, reward, next_state, done, steps):
        gamma = self.buffer.gamma
        q = self.softq.train_step(state, action)
        next_action, next_logpi = self.actor.train_step(next_state)
        next_q = self.target_softq.train_step(next_state, next_action)
        # for brevity, we don't compute n-th value and use n-th q directly here
        target_q = reward + gamma**steps * (1-done) * next_q
        
        priority = tf.abs(target_q - q)
        priority += self.per_epsilon
        priority **= self.per_alpha

        return priority
