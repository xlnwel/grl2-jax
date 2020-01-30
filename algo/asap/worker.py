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
from algo.asap.utils import *


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
        buffer_keys = ['state', 'action', 'reward', 'done', 'steps']
        buffer = buffer_fn(buffer_config, *buffer_keys)

        super().__init__(
            name=name,
            worker_id=worker_id,
            models=models,
            env=env,
            buffer=buffer,
            actor=models['actor'],
            value=models['q1'],
            config=config)

        self.weight_repo = {}                                 # map score to Weights
        self.mode = (Mode.LEARNING, Mode.EVOLUTION, Mode.REEVALUATION)
        self.mode_prob = np.array(self.mode_prob, dtype=np.float)
        self.mode_prob_backup = self.mode_prob
        self.raw_bookkeeping = BookKeeping('raw')
        self.best_score = -float('inf')
        self.info_to_print = []
        # self.reevaluation_bookkeeping = BookKeeping('reeval')

    def run(self, learner, replay):
        step = 0
        log_time = self.LOG_INTERVAL
        while step < self.MAX_STEPS:
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                mode, score, tag, weights, eval_times = self._choose_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, scores, epslens = self.eval_model(
                    weights, step, replay, 
                    evaluation=mode == Mode.REEVALUATION, 
                    tag=tag, store_exp=mode != Mode.REEVALUATION)
            eval_times = eval_times + self.n_envs

            if mode != Mode.REEVALUATION:
                with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                    self._send_data(replay)

            score += self.n_envs / eval_times * (np.mean(scores) - score)

            pop_score = store_weights(
                self.weight_repo, mode, score, tag, weights, 
                eval_times, self.REPO_CAP, fifo=self.FIFO,
                fitness_method=self.fitness_method, c=self.cb_c)

            status = self._make_decision(mode, score, tag, eval_times, pop_score)
            
            self._update_mode_prob()

            self.info_to_print = print_repo(self.weight_repo, self.model_name, c=self.cb_c, info=self.info_to_print)

            if self.env.name == 'BipedalWalkerHardcore-v2' and eval_times > 100 and score > 300:
                self.save()
            elif step > log_time and tag == Tag.EVOLVED:
                self.save(print_terminal_info=False)
                log_time += self.LOG_INTERVAL

    def _choose_weights(self, learner):
        if len(self.weight_repo) < self.MIN_EVOLVE_MODELS:
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
            weights, n = evolve_weights(
                self.weight_repo, 
                min_evolv_models=self.MIN_EVOLVE_MODELS, 
                max_evolv_models=self.MAX_EVOLVE_MODELS, 
                wa_selection=self.WA_SELECTION,
                wa_evolution=self.WA_EVOLUTION, 
                fitness_method=self.fitness_method,
                c=self.cb_c)
            self.info_to_print.append(((f'{self.name}_{self.id}: {n} models are used for evolution', ), 'blue'))
            score = 0
            eval_times = 0
        elif mode == Mode.REEVALUATION:
            w = fitness_from_repo(self.weight_repo, 'norm', c=self.cb_c)
            score = random.choices(list(self.weight_repo), weights=w)[0]
            tag = self.weight_repo[score].tag
            weights = self.weight_repo[score].weights
            eval_times = self.weight_repo[score].eval_times
            del self.weight_repo[score]
        
        return mode, score, tag, weights, eval_times

    def _make_decision(self, mode, score, tag, eval_times, pop_score):
        if score != pop_score:
            status = Status.ACCEPTED
        else:
            status = Status.REJECTED

        self.raw_bookkeeping.add(tag, status)

        self.info_to_print.append(((f'{self.name}_{self.id} Mode({mode}): {tag} model has been evaluated({eval_times}).', 
                                f'Score: {score:.3g}',
                                f'Decision: {status}'), 'green'))
        return status

    def _update_mode_prob(self):
        fracs = analyze_repo(self.weight_repo)
        mode_prob = np.zeros_like(self.mode_prob)
        # if self.env_name == 'BipedalWalkerHardcore-v2':
        #     if min(self.weight_repo) > 300:
        #         mode_prob[2] = 1
        #         mode_prob[0] = mode_prob[1] = 0
        #         self.mode_prob = mode_prob
        #         return
        #     elif min(self.weight_repo) > 295:
        #         mode_prob[2] = .5
        #         mode_prob[0] = mode_prob[1] = 0.25
        #         self.mode_prob = mode_prob
        #         return

        self.mode_prob = self.mode_prob_backup
        mode_prob[2] = self.REEVAL_PROB
        remain_prob = 1 - mode_prob[2] - self.MIN_LEARN_PROB - self.MIN_EVOLVE_PROB

        mode_prob[0] = self.MIN_LEARN_PROB + fracs['frac_learned'] * remain_prob
        mode_prob[1] = self.MIN_EVOLVE_PROB + fracs['frac_evolved'] * remain_prob

        self.mode_prob = self.mode_polyak * self.mode_prob + (1 - self.mode_polyak) * mode_prob
        self.mode_prob /= np.sum(self.mode_prob)    # renormalize so that probs sum to one
        self.mode_prob_backup = self.mode_prob
        np.testing.assert_allclose(np.sum(self.mode_prob), 1)

    def _log_condition(self):
        return self.logger.get_count('score') > 0 and self.logger.get_count('evolved_score') > 0

    def _logging(self, step):
        self.store(**self.get_value('score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('epslen', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('evolved_score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('evolved_epslen', mean=True, std=True, min=True, max=True))
        self.store(mode_learned=self.mode_prob[0], mode_evolved=self.mode_prob[1])
        self.store(**analyze_repo(self.weight_repo))
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
    config['mode_prob'] = [1, 0, 0]

    worker = Worker.remote(name, worker_id, model_fn, buffer_fn, config, 
                        model_config, env_config, buffer_config)

    ray.get(worker.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=buffer_config
    )))

    return worker
