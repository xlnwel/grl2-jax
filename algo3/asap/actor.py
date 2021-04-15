import random
import numpy as np

from core.tf_config import *
from utility.display import pwc
from utility.timer import Timer
from algo.apex.actor import get_learner_class, get_evaluator_class, get_worker_class as get_apex_worker_class
from algo3.asap.utils import *


def get_worker_class(AgentBase):
    WorkerBase = get_apex_worker_class(AgentBase)
    class Worker(WorkerBase):
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
            self._valid_mode = (Mode.LEARNING, Mode.EVOLUTION, Mode.REEVALUATION)
            self._mode_prob = [.6, .3, 0.1]
            self._raw_bookkeeping = BookKeeping('raw')
            self._tag = Tag.LEARNED
            # self.reevaluation_bookkeeping = BookKeeping('reeval')

        def run(self, learner, replay, monitor):
            while True:
                self._mode, score, self._tag, weights, eval_times = self._choose_weights(learner)
                self.model.set_weights(weights)
                self._run(replay)
                eval_times += self._record_envs
                score = (score + self._sum_score) / eval_times
                status = self._make_decision(self._mode, score, eval_times)
                if status == Status.ACCEPTED:
                    store_weights(
                        self._weight_repo, self._mode, score, self._tag, weights, 
                        eval_times, self.REPO_CAP, fifo=self.FIFO,
                        fitness_method=self._fitness_method, c=self._cb_c)
                pwc(f'{self._id} Mode({self._mode}): {self._tag} model has been evaluated({eval_times}).', 
                    f'Score: {score:.3g}',
                    f'Decision: {status}', color='green')
                print_repo(self._weight_repo, self._id)
                self._send_episode_info(monitor)
                
                self._update_mode_prob()

        def _process_input(self, obs, evaluation, env_output):
            evaluation = self._evaluation and self._valid_mode == Mode.REEVALUATION
            obs, kwargs = super()._process_input(obs, evaluation, env_output)
            kwargs['evaluation'] = evaluation
            return obs, kwargs

        def _choose_weights(self, learner):
            if len(self._weight_repo) < self.MIN_EVOLVE_MODELS:
                mode = Mode.LEARNING
            else:
                mode = random.choices(self._valid_mode, weights=self._mode_prob)[0]

            if mode == Mode.LEARNING:
                with Timer('pull_weights', 1000):
                    tag = Tag.LEARNED
                    weights = self._pull_weights(learner)
                    score = 0
                    eval_times = 0
            elif mode == Mode.EVOLUTION:
                with Timer('evolve_weights', 1000):
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
            if isinstance(score, (int, float)):
                if self._tag == Tag.LEARNED:
                    self._info['score'].append(score)
                    self._info['epslen'].append(epslen)
                else:
                    self._info['evolved_score'].append(score)
                    self._info['epslen'].append(epslen)
            else:
                if self._tag == Tag.LEARNED:
                    self._info['score'] += list(score)
                    self._info['epslen'] += list(epslen)
                else:
                    self._info['evolved_score'] += score
                    self._info['epslen'] += epslen
            self._sum_score = np.sum(score)

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

        def _send_episode_info(self, learner):
            if 'evolved_score' in self._info and self._weight_repo:
                learner.record_episode_info.remote(
                    **self._raw_bookkeeping.stats(),
                    **self._info, 
                    **analyze_repo(self._weight_repo),
                    learn_prob=self._mode_prob[0],
                    evolve_prob=self._mode_prob[1])
                self._info.clear()
                self._raw_bookkeeping.reset()
    return Worker
