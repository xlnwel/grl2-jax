from collections import namedtuple
import random
import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_gym_env
from algo.apex.buffer import create_local_buffer
from algo.apex.base_worker import BaseWorker
from algo.apex_es.utils import *


Weights = namedtuple('Weights', 'tag weights')

LOG_STEPS = 10000

@ray.remote(num_cpus=1)
class Worker(BaseWorker):
    """ Interface """
    def __init__(self, 
                name,
                worker_id, 
                model_fn,
                buffer_fn,
                config,
                model_config, 
                env_config, 
                buffer_config):
        tf_config.configure_threads(1, 1)
        tf_config.configure_gpu()

        env = create_gym_env(env_config)
        
        models = Ensemble(model_fn, model_config, env.state_shape, env.action_dim, env.is_action_discrete)
        
        buffer_config['seqlen'] = env.max_episode_steps
        buffer = buffer_fn(
            buffer_config, env.state_shape, 
            env.state_dtype, env.action_shape, 
            env.action_dtype, config['gamma'])

        super().__init__(
            name=name,
            worker_id=worker_id,
            models=models,
            env=env,
            buffer=buffer,
            actor=models['actor'],
            value=models['q1'],
            config=config)

        self.store_map = {}                                 # map score to Weights
        self.mode = (Mode.LEARNING, Mode.EVOLUTION, Mode.REEVALUATION)
        self.raw_bookkeeping = BookKeeping('raw')
        self.best_score = -float('inf')
        # self.reevaluation_bookkeeping = BookKeeping('reeval')

    def run(self, learner, replay):
        step = 0
        while step < self.MAX_STEPS:
            self.set_summary_step(step)
            with TBTimer(f'{self.name} pull weights', self.TIME_PERIOD, to_log=self.timer):
                mode, tag, weights = self._choose_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_PERIOD, to_log=self.timer):
                step, scores, epslens = self.eval_model(weights, step, replay)

            with TBTimer(f'{self.name} send data', self.TIME_PERIOD, to_log=self.timer):
                self._send_data(replay)

            score = np.mean(scores)
            
            status = self._make_decision(score)
            self.raw_bookkeeping.add(tag, status)

            pwc(f'{self.name}_{self.id}: {tag} model has been evaluated.', 
                f'Score: {score:.3g}',
                f'Decision: {status}', color='green')

            if status == Status.ACCEPTED:
                self._store_weights(score, tag, weights)

            self._periodic_logging(step)

    def _send_data(self, replay):
        """ sends data to replay """
        mask, data = self.buffer.sample()
        data_tesnors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        if not self.replay_type.endswith('uniform'):
            data['priority'] = self.compute_priorities(**data_tesnors).numpy()
        
        # squeeze since many terms in data is of shape [None, 1]
        for k, v in data.items():
            data[k] = np.squeeze(v)

        replay.merge.remote(data, data['state'].shape[0])

        self.buffer.reset()

    def _choose_weights(self, learner):
        if len(self.store_map) < self.MIN_EVOLVE_MODELS:
            mode = Mode.LEARNING
        else:
            mode = random.choices(self.mode, weights=self.mode_prob)[0]

        if mode == Mode.LEARNING:
            tag = Tag.LEARNED
            weights = self.pull_weights(learner)
        elif mode == Mode.EVOLUTION:
            tag = Tag.EVOLVED
            weights, n = evolve_weights(self.store_map, min_evolv_models=self.MIN_EVOLVE_MODELS)
            pwc(f'{self.name}_{self.id}: {n} models are used for evolution', color='blue')
        elif mode == Mode.REEVALUATION:
            scores = sorted(list(self.store_map.keys()), reverse=True)
            if len(scores) >= self.min_reeval_models:
                # starts from the best model in store, we search for a model that has not been evaluated
                for score in scores:
                    tag, weights = self.store_map[score]
                    if tag == Tag.REEVALUATED:
                        continue
                    else:
                        del self.store_map[score]
                        return mode, tag, weights
            tag = Tag.EVOLVED
            weights = self._evolve_weights()
        
        return mode, tag, weights

    def _make_decision(self, score):
        self.best_score = max(score, self.best_score)
        min_score = min(self.store_map.keys()) if self.store_map else -float('inf')
        if score > min_score:
            status = Status.ACCEPTED
        # elif score > min_score - self.SLACK:
        #     status = Status.TOLERATED
        else:
            status = Status.REJECTED
        
        return status

    def _store_weights(self, score, tag, weights):
        self.store_map[score] = Weights(tag, weights)
        while self.store_map and self.best_score - min(self.store_map) > self.SLACK:
            remove_worst_weights(self.store_map)
        self._print_store()

    def _periodic_logging(self, step):
        if step > self.log_steps:
            # record stats
            self.store(**self.raw_bookkeeping.stats())
            # self.store(**self.reevaluation_bookkeeping.stats())
            self.raw_bookkeeping.reset()
            # self.reevaluation_bookkeeping.reset()
            self.log(step, print_terminal_info=False)
            self.log_steps += self.LOG_STEPS

    def _print_store(self):
        store = [(score, weights.tag) for score, weights in self.store_map.items()]
        store = sorted(store, key=lambda x: x[0], reverse=True)
        pwc(f"{self.name}_{self.id}: current stored models", 
            f"{[f'({x[0]:.3g}, {x[1]})' for x in store]}", 
            color='magenta')

def create_worker(name, worker_id, model_fn, config, model_config, 
                env_config, buffer_config):
    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    buffer_config = buffer_config.copy()

    buffer_config['n_envs'] = env_config.get('n_envs', 1)
    buffer_fn = create_local_buffer

    env_config['seed'] = worker_id
    
    config['model_name'] = f'worker_{worker_id}'
    config['mode_prob'] = [1-.2*worker_id, .2*worker_id, 0]
    config['TIME_PERIOD'] = 1000
    config['LOG_STEPS'] = 1e5
    config['MAX_STEPS'] = int(1e8)
    config['MIN_EVOLVE_MODELS'] = 2
    config['SLACK'] = 10
    config['replay_type'] = buffer_config['type']

    worker = Worker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker
