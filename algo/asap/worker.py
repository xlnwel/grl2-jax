import random
import numpy as np
import tensorflow as tf
import ray

from core import tf_config
from core.ensemble import Ensemble
from utility.display import pwc
from utility.timer import TBTimer
from utility.utils import step_str
from env.gym_env import create_gym_env
from algo.apex.buffer import create_local_buffer
from algo.apex.base_worker import BaseWorker
from algo.asap.utils import *


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
        buffer_config['seqlen'] = env.max_episode_steps
        buffer = buffer_fn(buffer_config)

        models = Ensemble(model_fn, model_config, env.state_shape, env.action_dim, env.is_action_discrete)

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
        self.raw_bookkeeping = BookKeeping('raw')
        self.info_to_print = []
        # self.reevaluation_bookkeeping = BookKeeping('reeval')

    def run(self, learner, replay):
        def eval_send(score, weights, tag, step, eval_times, evaluation, store_data):
            step, scores, epslens = self.eval_model(
                weights, step, tag=tag, evaluation=evaluation, store_data=store_data)
            
            eval_times += self.n_envs
            
            self._log_episodic_info(tag, scores, epslens)

            score += self.n_envs / eval_times * (np.mean(scores) - score)

            if store_data:
                self._send_data(replay, tag=tag)

            return score, eval_times, step

        step = 0
        log_time = self.LOG_INTERVAL
        while step < self.MAX_STEPS:
            mode, score, tag, weights, eval_times = self._choose_weights(learner)

            is_reeval = mode == Mode.REEVALUATION
            score, eval_times, step = eval_send(
                score, weights, tag, step, eval_times, 
                evaluation=is_reeval, store_data=not is_reeval)

            if (len(self.weight_repo) < self.REPO_CAP or score > min(self.weight_repo)):
                score, eval_times, step = eval_send(
                    score, weights, tag, step, eval_times, 
                    evaluation=True, store_data=False)

            status = self._make_decision(mode, score, tag, eval_times, step)

            if status == Status.ACCEPTED:
                store_weights(
                    self.weight_repo, mode, score, tag, weights, 
                    eval_times, self.REPO_CAP, fifo=self.FIFO,
                    fitness_method=self.fitness_method, c=self.cb_c)
            
            self._update_mode_prob()

            self.info_to_print = print_repo(self.weight_repo, self.model_name, c=self.cb_c, info=self.info_to_print)

            if step > log_time:
                self.set_weights(self.weight_repo[max(self.weight_repo)].weights)
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
            score = 0
            eval_times = 0
            if random.random() < self.EVOLVE_LEARN_PROB:
                learned_weights = self.pull_weights(learner)
                weights = [weights, learned_weights]
                weights = average_weights(weights)
        elif mode == Mode.REEVALUATION:
            w = fitness_from_repo(self.weight_repo, 'norm')
            score = random.choices(list(self.weight_repo.keys()), weights=w)[0]
            tag = self.weight_repo[score].tag
            weights = self.weight_repo[score].weights
            eval_times = self.weight_repo[score].eval_times
            del self.weight_repo[score]
        
        return mode, score, tag, weights, eval_times

    def _log_episodic_info(self, tag, scores, epslens):
        if scores is not None:
            if tag == 'Learned':
                self.store(
                    score=scores,
                    epslen=epslens,
                )
            else:
                self.store(
                    evolved_score=scores,
                    evolved_epslen=epslens,
                )

    def _make_decision(self, mode, score, tag, eval_times, step):
        if len(self.weight_repo) < self.REPO_CAP or score > min(self.weight_repo):
            status = Status.ACCEPTED
        else:
            status = Status.REJECTED

        self.raw_bookkeeping.add(tag, status)

        self.info_to_print.append((
            (f'{self.name}_{self.id} Mode({mode}) Step({step_str(step)}): {tag} model has been evaluated({eval_times}).', 
            f'Score: {score:.3g}',
            f'Decision: {status}'), 'green'))
            
        return status

    def _update_mode_prob(self):
        fracs = analyze_repo(self.weight_repo)
        if self.env.name == 'BipedalWalkerHardcore-v2' and min(self.weight_repo) > 300:
            self.mode_prob[2] = 1
            self.mode_prob[0] = self.mode_prob[1] = 0
        else:
            self.mode_prob[2] = self.REEVAL_PROB
            remain_prob = 1 - self.mode_prob[2] - self.MIN_LEARN_PROB - self.MIN_EVOLVE_PROB

            self.mode_prob[0] = self.MIN_LEARN_PROB + fracs['frac_learned'] * remain_prob
            self.mode_prob[1] = self.MIN_EVOLVE_PROB + fracs['frac_evolved'] * remain_prob
        np.testing.assert_allclose(sum(self.mode_prob), 1)
        self.info_to_print.append((
            (f'mode prob: {self.mode_prob[0]:3g}, {self.mode_prob[1]:3g}, {self.mode_prob[2]:3g}', ), 'blue'
        ))

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
