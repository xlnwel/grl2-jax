import time
import threading
import functools
import collections
import numpy as np
import ray

from core.tf_config import *
from core.decorator import config, override
from utility.display import pwc
from utility.utils import Every
from utility.ray_setup import cpu_affinity
from utility.run import Runner, evaluate, RunMode
from utility import pkg
from env.func import create_env
from core.dataset import process_with_env, RayDataset
    

def get_base_learner_class(BaseAgent):
    class BaseLearner(BaseAgent):            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()
            
        def _learning(self):
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
            
            while not ray.get(replay.good_to_learn.remote()):
                time.sleep(1)
            data = ray.get(replay.sample.remote())
            data_format = {k: (v.shape, v.dtype) for k, v in data.items()}
            print('data format')
            for k, v in data_format.items():
                print('\t', k, v)
            one_hot_action = config.get('one_hot_action', True)
            process = functools.partial(process_with_env, 
                env=env, one_hot_action=one_hot_action)
            dataset = RayDataset(replay, data_format, process)

            model = model_fn(
                config=model_config, 
                env=env)

            super().__init__(
                name='learner',
                config=config, 
                models=model,
                dataset=dataset,
                env=env,
            )
            
    return Learner


def get_worker_class(BaseAgent):
    class Worker(BaseAgent):
        """ Initialization """
        def __init__(self,
                    *,
                    config,
                    worker_id,
                    model_config, 
                    env_config, 
                    buffer_config,
                    model_fn,
                    buffer_fn):
            silence_tf_logs()
            configure_threads(1, 1)
            configure_gpu()
            configure_precision(config.get('precision', 32))
            self._id = worker_id

            self.env = create_env(env_config)
            self.n_envs = self.env.n_envs

            self.buffer = buffer_fn(buffer_config)

            models = model_fn( 
                config=model_config, 
                env=self.env)

            super().__init__(
                name=f'worker_{worker_id}',
                config=config,
                models=models,
                dataset=self.buffer,
                env=self.env)
            
            self._run_mode = getattr(self, '_run_mode', RunMode.NSTEPS)
            assert self._run_mode in [RunMode.NSTEPS, RunMode.TRAJ]
            self.runner = Runner(
                self.env, self, 
                nsteps=self.SYNC_PERIOD if self._run_mode == RunMode.NSTEPS else None,
                run_mode=self._run_mode)

            self._return_stats = self._worker_side_prioritization \
                or buffer_config.get('max_steps', 0) > buffer_config.get('n_steps', 1)
            collect_fn = pkg.import_module('agent', algo=self._algorithm, place=-1).collect
            self._collect = functools.partial(collect_fn, self.buffer)

            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]
            self._info = collections.defaultdict(list)
            
            print(f'{worker_id} action epsilon:', self._act_eps)
            if hasattr(self.model, 'actor'):
                print(f'{worker_id} action inv_temp:', np.squeeze(self.model.actor.act_inv_temp))

        """ Call """
        def _process_input(self, obs, evaluation, env_output):
            obs, kwargs = super()._process_input(obs, evaluation, env_output)
            kwargs['return_stats'] = self._return_stats
            return obs, kwargs

        """ Worker Methods """
        def prefill_replay(self, replay):
            while not ray.get(replay.good_to_learn.remote()):
                self._run(replay)

        def run(self, learner, replay, monitor):
            while True:
                weights = self._pull_weights(learner)
                self.model.set_weights(weights)
                self._run(replay)
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

        def _run(self, replay):
            def collect(*args, **kwargs):
                self._collect(*args, **kwargs)
                if self.buffer.is_full():
                    self._send_data(replay)
            start_step = self.runner.step
            end_step = self.runner.run(step_fn=collect)
            return end_step - start_step

        def _send_data(self, replay, buffer=None):
            buffer = buffer or self.buffer
            data = buffer.sample()

            if self._worker_side_prioritization:
                data['priority'] = self._compute_priorities(**data)
            data.pop('q', None)
            data.pop('next_q', None)
            replay.merge.remote(data, data['action'].shape[0])
            buffer.reset()

        def _send_episode_info(self, learner):
            if self._info:
                learner.record_episode_info.remote(self._id, **self._info)
                self._info.clear()

        def _compute_priorities(self, reward, discount, steps, q, next_q, **kwargs):
            target_q = reward + discount * self._gamma**steps * next_q
            priority = np.abs(target_q - q)
            priority += self._per_epsilon
            priority **= self._per_alpha

            return priority
    
    return Worker


def get_evaluator_class(BaseAgent):
    class Evaluator(BaseAgent):
        """ Initialization """
        def __init__(self, 
                    *,
                    config,
                    name='evaluator',
                    model_config,
                    env_config,
                    model_fn):
            silence_tf_logs()
            configure_threads(1, 1)

            env_config.pop('reward_clip', False)
            self.env = env = create_env(env_config)
            self.n_envs = self.env.n_envs

            model = model_fn(
                    config=model_config, 
                    env=env)
            
            super().__init__(
                name='learner',
                config=config, 
                models=model,
                dataset=None,
                env=env,
            )
        
            if not hasattr(self, '_pull_names'):
                self._pull_names = [k for k in self.model.keys() if 'target' not in k]
            self._info = collections.defaultdict(list)

        """ Evaluator Methods """
        def run(self, learner, monitor):
            step = 0
            if getattr(self, 'RECORD_PERIOD', False):
                to_record = Every(self.RECORD_PERIOD)
            else:
                to_record = lambda x: False 
            while True:
                step += 1
                weights = self._pull_weights(learner)
                self.model.set_weights(weights)
                self._run(record=to_record(step))
                self._send_episode_info(monitor)

        def _pull_weights(self, learner):
            return ray.get(learner.get_weights.remote(name=self._pull_names))

        def _run(self, record):        
            score, epslen, video = evaluate(self.env, self, 
                record=record, n=self.N_EVALUATION)
            self.store(score, epslen, video)

        def store(self, score, epslen, video):
            self._info['eval_score'] += score
            self._info['eval_epslen'] += epslen
            if video is not None:
                self._info['video'] = video

        def _send_episode_info(self, learner):
            if self._info:
                learner.record_episode_info.remote(**self._info)
                self._info.clear()
    
    return Evaluator
