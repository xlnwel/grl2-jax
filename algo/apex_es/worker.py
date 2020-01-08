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
        self.info_to_print = []
        # self.reevaluation_bookkeeping = BookKeeping('reeval')

    def run(self, learner, replay):
        step = 0
        i = 0
        while step < self.MAX_STEPS:
            i += 1
            self.set_summary_step(step)
            
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                mode, score, tag, weights, eval_times = self._choose_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, scores, epslens = self.eval_model(weights, step, replay)
            eval_times = eval_times + self.n_envs

            with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                self._send_data(replay)

            score += self.n_envs / eval_times * (np.mean(scores) - score)
            
            status = self._make_decision(score, tag)

            if status == Status.ACCEPTED:
                store_weights(self.store_map, score, tag, weights, eval_times, self.STORE_CAP)
            
            if self.id != 0:
                self._update_mode_prob()

            self.info_to_print = print_store(self.store_map, self.model_name, self.info_to_print)
            self._periodic_logging(step, i)

    def _choose_weights(self, learner):
        if len(self.store_map) < self.MIN_EVOLVE_MODELS:
            mode = Mode.LEARNING
        else:
            mode = random.choices(self.mode, weights=self.mode_prob)[0]

        if mode == Mode.LEARNING:
            tag = Tag.LEARNED
            weights = self.pull_weights(learner)
            score = 0
            eval_times = 0
        elif mode == Mode.EVOLUTION:
            tag = Tag.EVOLVED
            weights, n = evolve_weights(self.store_map)
            self.info_to_print.append(((f'{self.name}_{self.id}: {n} models are used for evolution', ), 'blue'))
            score = 0
            eval_times = 0
        elif mode == Mode.REEVALUATION:
            score = random.choices(list(self.store_map.keys()))[0]
            tag = self.store_map[score].tag
            weights = self.store_map[score].weights
            eval_times = self.store_map[score].eval_times
            del self.store_map[score]
        
        return mode, score, tag, weights, eval_times

    def _make_decision(self, score, tag):
        if len(self.store_map) < self.STORE_CAP:
            status = Status.ACCEPTED
        else:
            min_score = min(self.store_map.keys()) if self.store_map else -float('inf')
            if score > min_score:
                status = Status.ACCEPTED
            else:
                status = Status.REJECTED
        self.raw_bookkeeping.add(tag, status)

        self.info_to_print.append(((f'{self.name}_{self.id}: {tag} model has been evaluated.', 
                                f'Score: {score:.3g}',
                                f'Decision: {status}'), 'green'))
        return status

    def _update_mode_prob(self):
        fracs = analyze_store(self.store_map)
        self.mode_prob[0] = 0.1 + fracs['frac_learned'] * .5
        self.mode_prob[1] = 0.1 + fracs['frac_evolved'] * .5

    def _periodic_logging(self, step, i):
        if i % self.LOG_INTERVAL == 0:
            self.store(mode_learned=self.mode_prob[0], mode_evolved=self.mode_prob[1])
            self.store(**analyze_store(self.store_map))
            # record stats
            self.store(**self.raw_bookkeeping.stats())
            # self.store(**self.reevaluation_bookkeeping.stats())
            self.raw_bookkeeping.reset()
            # self.reevaluation_bookkeeping.reset()
            self.log(step, print_terminal_info=False)

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
    config['replay_type'] = buffer_config['type']
    if worker_id == 0:
        config['mode_prob'] = [.7, 0, .3]
    else:
        config['mode_prob'] = [.5, .1, .3]

    worker = Worker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker
