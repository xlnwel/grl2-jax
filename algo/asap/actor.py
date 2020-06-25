import random
import collections
import numpy as np
import tensorflow as tf
import ray

from core.tf_config import *
from core.module import Ensemble
from utility.display import pwc
from env.gym_env import create_env
from algo.apex.actor import Worker as BaseWorker
from algo.apex.actor import get_learner_class, get_evaluator_class
from algo.asap.utils import *


class Worker(BaseWorker):
    def __init__(self, 
                *,
                worker_id,
                config, 
                model_config, 
                env_config, 
                buffer_config,
                model_fn,
                buffer_fn):
        super().__init__(
            worker_id=worker_id,
            config=config,
            model_config=model_config,
            env_config=env_config,
            buffer_config=buffer_config,
            model_fn=model_fn,
            buffer_fn=buffer_fn)

        self._weight_repo = {}                                 # map score to Weights
        self._mode = (Mode.LEARNING, Mode.EVOLUTION, Mode.REEVALUATION)
        self._mode_prob = (.6, .3, 0.1)
        self._raw_bookkeeping = BookKeeping('raw')
        self._tag = Tag.LEARNED
        # self.reevaluation_bookkeeping = BookKeeping('reeval')

    def run(self, learner, replay):
        while True:
            mode, score, self._tag, weights, eval_times = self._choose_weights(learner)
            self._run(weights, replay)
            eval_times += 1
            curr_score = self._info['score'][-1] if self._tag == Tag.LEARNED \
                else self._info['evolved_score'][-1]
            score = (score + curr_score) / eval_times
            status = self._make_decision(mode, score, eval_times)
            if status == Status.ACCEPTED:
                store_weights(
                    self._weight_repo, mode, score, self._tag, weights, 
                    eval_times, self.REPO_CAP, fifo=self.FIFO,
                    fitness_method=self._fitness_method, c=self._cb_c)
            pwc(f'{self._id} Mode({mode}): {self._tag} model has been evaluated({eval_times}).', 
                f'Score: {score:.3g}',
                f'Decision: {status}', color='green')
            print_repo(self._weight_repo, self._id)
            self._send_episode_info(learner)
            
            # self._update_mode_prob()

    def _choose_weights(self, learner):
        if len(self._weight_repo) < self.MIN_EVOLVE_MODELS:
            mode = Mode.LEARNING
        else:
            mode = random.choices(self._mode, weights=self._mode_prob)[0]

        if mode == Mode.LEARNING:
            tag = Tag.LEARNED
            weights = self._pull_weights(learner)
            score = 0
            eval_times = 0
        elif mode == Mode.EVOLUTION:
            tag = Tag.EVOLVED
            weights, n = evolve_weights(
                self._weight_repo, 
                min_evolv_models=self.MIN_EVOLVE_MODELS, 
                max_evolv_models=self.MAX_EVOLVE_MODELS, 
                wa_selection=self.WA_SELECTION,
                wa_evolution=self.WA_EVOLUTION, 
                fitness_method=self._fitness_method,
                c=self._cb_c)
            score = 0
            eval_times = 0
        elif mode == Mode.REEVALUATION:
            w = fitness_from_repo(self._weight_repo, 'norm')
            score = random.choices(list(self._weight_repo.keys()), weights=w)[0]
            tag = self._weight_repo[score].tag
            weights = self._weight_repo[score].weights
            eval_times = self._weight_repo[score].eval_times
            del self._weight_repo[score]
        
        return mode, score, tag, weights, eval_times

    def store(self, score, epslen):
        if self._tag == Tag.LEARNED:
            self._info['score'].append(score)
            self._info['epslen'].append(epslen)
        else:
            self._info['evolved_score'].append(score)
            self._info['evolved_epslen'].append(epslen)

    def _make_decision(self, mode, score, eval_times):
        if len(self._weight_repo) < self.REPO_CAP or score > min(self._weight_repo):
            status = Status.ACCEPTED
        else:
            status = Status.REJECTED

        self._raw_bookkeeping.add(self._tag, status)
            
        return status

    def _update_mode_prob(self):
        fracs = analyze_repo(self._weight_repo)
        # if self.env.name == 'BipedalWalkerHardcore-v2' and min(self._weight_repo) > 300:
        #     self._mode_prob[2] = 1
        #     self._mode_prob[0] = self._mode_prob[1] = 0
        # else:
        self._mode_prob[2] = self.REEVAL_PROB
        remain_prob = 1 - self._mode_prob[2] - self.MIN_LEARN_PROB - self.MIN_EVOLVE_PROB

        self._mode_prob[0] = self.MIN_LEARN_PROB + fracs['frac_learned'] * remain_prob
        self._mode_prob[1] = self.MIN_EVOLVE_PROB + fracs['frac_evolved'] * remain_prob
        np.testing.assert_allclose(sum(self._mode_prob), 1)
        self.info_to_print.append((
            (f'mode prob: {self._mode_prob[0]:3g}, {self._mode_prob[1]:3g}, {self._mode_prob[2]:3g}', ), 'blue'
        ))

    def _send_episode_info(self, learner):
        if self._info and self._weight_repo:
            learner.record_episode_info.remote(
                **self._raw_bookkeeping.stats(),
                **self._info, 
                **analyze_repo(self._weight_repo))
            self._info.clear()
            self._raw_bookkeeping.reset()

def get_worker_class():
    return Worker