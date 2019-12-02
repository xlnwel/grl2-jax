import threading
import time
import numpy as np
import tensorflow as tf
import ray

from core.tf_config import configure_gpu, configure_threads
from utility.display import pwc
from utility.timer import Timer
from env.gym_env import create_gym_env
from replay.func import create_replay
from replay.data_pipline import Dataset
from algo.sac.nn import create_model
from algo.sac.agent import Agent


TRAIN_TIME_PERIOD = 1000
STEP_TIME_PERIOD = 20000

def create_learner(name, config, model_config, env_config, buffer_config):
    @ray.remote(num_gpus=0.3, num_cpus=2)
    class Learner(Agent):
        """ Interface """
        def __init__(self,
                    name, 
                    config, 
                    model_config,
                    env_config,
                    buffer_config):
            # tf.debugging.set_log_device_placement(True)
            configure_threads(2, 2)
            configure_gpu()
 
            env = create_gym_env(env_config)
            buffer_keys = ['state', 'action', 'reward', 'done', 'steps']
            self.buffer = create_replay(buffer_config, *buffer_keys, env.state_shape)
            dataset = Dataset(buffer, env.state_shape, env.state_dtype, env.action_shape, env.action_dtype)
            models = create_model(model_config, env.state_shape, env.action_shape, env.is_action_discrete)
            
            self.n_workers = config['n_workers']
            # we don't use a queue here since we want to retrieve all states at once
            # and ray is thread safe in nature
            self.state_queue = []   

            super().__init__(
                name=name, 
                config=config, 
                models=models,
                dataset=dataset,
                state_shape=env.state_shape,
                state_dtype=env.state_dtype,
                action_shape=env.action_shape,
                action_dtype=env.action_dtype,
            )

            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()

        def start_action_loop(self, workers):
            self._action_thread = threading.Thread(
                target=self._action_loop, args=[workers], daemon=True)
            self._action_thread.start()

        def enqueue_state(self, worker_id, env_id, state):
            with Timer(f'{self.name}: enqueue_states', 10 * STEP_TIME_PERIOD):
                self.state_queue.append((worker_id, env_id, state))

        def add_transition(self, state, action, reward, done, next_state):
            self.buffer.add(state, action, reward, done, next_state)

        def _learning(self):
            while not self.dataset.good_to_learn():
                time.sleep(1)
            pwc('Learner start learning...', color='blue')
            step = 0
            self.writer.set_as_default()
            while True:
                step += 1
                with Timer('learn_log', TRAIN_TIME_PERIOD):
                    self.learn_log()
                if step % 1000 == 0:
                    self.log_summary(self.logger.get_stats(), step)
                    self.save(step)
        
        def _action_loop(self, workers):
            # TODO: consider making an actor for this method
            pwc('Action loop starts...', color='blue')
            while True:
                with Timer(f'{self.name} dequeue', TRAIN_TIME_PERIOD):
                    while len(self.state_queue) < self.n_workers:
                        # pwc('Learner is going to sleep', color='magenta')
                        time.sleep(.01)
                    worker_ids, env_ids, states = list(zip(*self.state_queue))
                    self.state_queue = []
                
                with Timer(f'{self.name} action', STEP_TIME_PERIOD):
                    actions = self.actor.step(tf.convert_to_tensor(states, tf.float32))
                [workers[wid].enqueue_action.remote(eid, a) 
                    for wid, eid, a in 
                    zip(worker_ids, env_ids, actions.numpy())]

    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    config['model_name'] = 'learner'
    env_config['n_workers'] = env_config['n_envs'] = 1
    
    learner = Learner.remote(name, config, model_config, env_config, buffer_config)

    learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        buffer=buffer_config
    ))

    return learner