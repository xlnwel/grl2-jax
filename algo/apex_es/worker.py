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
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                mode, tag, weights = self._choose_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, scores, epslens = self.eval_model(weights, step, replay)

            with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                self._send_data(replay)

            score = np.mean(scores)
            
            status = self._make_decision(score)
            self.raw_bookkeeping.add(tag, status)

            pwc(f'{self.name}_{self.id}: {tag} model has been evaluated.', 
                f'Score: {score:.3g}',
                f'Decision: {status}', color='green')

            if status == Status.ACCEPTED:
                self._store_weights(score, tag, weights)

            print_store(self.store_map, self.model_name)
            
            self._periodic_logging(step)

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
        while len(self.store_map) > self.store_cap:
            remove_worst_weights(self.store_map)

    def _periodic_logging(self, step):
        if step > self.log_steps:
            self.store(**analyze_store(self.store_map))
            # record stats
            self.store(**self.raw_bookkeeping.stats())
            # self.store(**self.reevaluation_bookkeeping.stats())
            self.raw_bookkeeping.reset()
            # self.reevaluation_bookkeeping.reset()
            self.log(step, print_terminal_info=False)
            self.log_steps += self.LOG_INTERVAL

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

    env_config['seed'] += worker_id * 100
    
    config['model_name'] = f'worker_{worker_id}'
    config['mode_prob'] = [1-.2*worker_id, .2*worker_id, 0]
    # config['SLACK'] = 10
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
