import random
import threading
from collections import namedtuple
import tensorflow as tf
import ray

from core.module import Ensemble
from core.tf_config import *
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_gym_env
from replay.data_pipline import DataFormat, RayDataset
from algo.apex.actor import BaseWorker
from algo.asap.utils import *



def get_learner_class(BaseAgent):
    class Learner(BaseAgent):
        """ Interface """
        def __init__(self,
                    name, 
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config):
            silence_tf_logs()
            configure_threads(1, 1)
            configure_gpu()

            self.env_name = env_config['name']
            env = create_gym_env(env_config)
            data_format = dict(
                state=DataFormat((None, *env.state_shape), env.state_dtype),
                action=DataFormat((None, *env.action_shape), env.action_dtype),
                reward=DataFormat((None, ), tf.float32), 
                next_state=DataFormat((None, *env.state_shape), env.state_dtype),
                done=DataFormat((None, ), tf.float32),
            )
            if replay.n_steps > 1:
                data_format['steps'] = DataFormat((None, ), tf.float32)
            dataset = RayDataset(replay, data_format)
            self.model = Ensemble(
                model_fn=model_fn, 
                config=model_config, 
                action_dim=env.action_dim, 
                is_action_discrete=env.is_action_discrete)
            
            super().__init__(
                name=name, 
                config=config, 
                models=self.model,
                dataset=dataset,
                env=env,
            )

            self.weight_repo = {}                                 # map score to Weights
            self.records = {}
            self.mode = (Mode.LEARNING, Mode.EVOLUTION, Mode.REEVALUATION)
            self.mode_prob = np.array(self.mode_prob, dtype=np.float)
            self.mode_prob_backup = self.mode_prob
            self.bookkeeping = BookKeeping('raw')
            
        def start_learning(self):
            self._learning_thread = threading.Thread(target=self._learning, daemon=True)
            self._learning_thread.start()

        def _learning(self):
            pwc(f'{self.name} starts learning...', color='blue')
            step = 0
            self._writer.set_as_default()
            while True:
                step += 1
                with TBTimer(f'{self.name} train', 10000, to_log=self.timer):
                    self.learn_log(step)
                if self.weight_repo and step % 1000 == 0:
                    print_repo(self.weight_repo, self._model_name, self.cb_c)
                    self.store(mode_learned=self.mode_prob[0], mode_evolved=self.mode_prob[1])
                    self.store(**analyze_repo(self.weight_repo))
                    self.store(**self.bookkeeping.stats())
                    self.bookkeeping.reset()
                    self.log(step, print_terminal_info=False)
                if step % 100000 == 0:
                    self.save(print_terminal_info=False)

        def get_weights(self, worker_id, name=None):
            mode, score, tag, weights, eval_times = self._choose_weights(worker_id, name=name)
            
            self.records[worker_id] = Records(mode, tag, weights, eval_times)

            return mode, score, tag, weights, eval_times

        def store_weights(self, worker_id, score, eval_times):
            mode, tag, weights, _ = self.records[worker_id]

            status = self._make_decision(score, tag, eval_times)

            if status == Status.ACCEPTED:
                store_weights(
                    self.weight_repo, mode, score, tag, weights, 
                    eval_times, self.REPO_CAP, fifo=self.FIFO,
                    fitness_method=self.fitness_method, c=self.cb_c)
            self._update_mode_prob()

        def _choose_weights(self, worker_id, name=None):
            if len(self.weight_repo) < self.MIN_EVOLVE_MODELS:
                mode = Mode.LEARNING
            else:
                mode = random.choices(self.mode, weights=self.mode_prob)[0]
            
            if mode == Mode.LEARNING:
                tag = Tag.LEARNED
                weights = self.model.get_weights(name=name)
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
            elif mode == Mode.REEVALUATION:
                w = fitness_from_repo(self.weight_repo, 'norm', c=self.cb_c)
                score = random.choices(list(self.weight_repo), weights=w)[0]
                tag = self.weight_repo[score].tag
                weights = self.weight_repo[score].weights
                eval_times = self.weight_repo[score].eval_times
                del self.weight_repo[score]
            else:
                raise ValueError(f'Unknown mode: {mode}')
        
            return mode, score, tag, weights, eval_times
        
        def _make_decision(self, score, tag, eval_times):
            if len(self.weight_repo) < self.REPO_CAP:
                status = Status.ACCEPTED
            else:
                min_score = min(self.weight_repo.keys())
                if score > min_score:
                    status = Status.ACCEPTED
                else:
                    status = Status.REJECTED

            self.bookkeeping.add(tag, status)

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

    return Learner

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
        super().__init__(
            config=config,
            name=name,
            worker_id=worker_id,
            model_fn=model_fn,
            buffer_fn=buffer_fn,
            model_config=model_config,
            env_config=env_config,
            buffer_config=buffer_config)

    def run(self, learner, replay):
        step = 0
        log_time = self.LOG_INTERVAL
        while step < self.MAX_STEPS:
            with TBTimer(f'{self.name} pull weights', self.TIME_INTERVAL, to_log=self.timer):
                mode, score, tag, weights, eval_times = self.pull_weights(learner)

            with TBTimer(f'{self.name} eval model', self.TIME_INTERVAL, to_log=self.timer):
                step, scores, epslens = self._run(
                    weights, step, replay, tag=tag, store_data=mode != Mode.REEVALUATION)
            eval_times = eval_times + self.n_envs

            self._log_episodic_info(tag, scores, epslens)

            # if mode != Mode.REEVALUATION:
            with TBTimer(f'{self.name} send data', self.TIME_INTERVAL, to_log=self.timer):
                self._send_data(replay, tag=tag)

            score += self.n_envs / eval_times * (np.mean(scores) - score)

            learner.store_weights.remote(self.id, score, eval_times)
            
            if self.env.name == 'BipedalWalkerHardcore-v2' and eval_times > 100 and score > 300:
                self.save(print_terminal_info=False)
            elif step > log_time and tag == Tag.EVOLVED:
                self.save(print_terminal_info=False)
                log_time += self.LOG_INTERVAL

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

    def _log_condition(self):
        return self._logger.get_count('score') > 0 and self._logger.get_count('evolved_score') > 0

    def _logging(self, step):
        # record stats
        self.store(**self.get_value('score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('epslen', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('evolved_score', mean=True, std=True, min=True, max=True))
        self.store(**self.get_value('evolved_epslen', mean=True, std=True, min=True, max=True))
        self.log(step, print_terminal_info=False)