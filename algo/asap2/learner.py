import random
import threading
from collections import namedtuple
import tensorflow as tf
import ray

from core.ensemble import Ensemble
from core.tf_config import configure_gpu, configure_threads
from utility.display import pwc
from utility.timer import TBTimer
from env.gym_env import create_gym_env
from replay.data_pipline import RayDataset
from algo.asap.utils import *



def create_learner(BaseAgent, name, model_fn, replay, config, model_config, env_config, replay_config):
    class Learner(BaseAgent):
        """ Interface """
        def __init__(self,
                    name, 
                    model_fn,
                    replay,
                    config, 
                    model_config,
                    env_config):
            # tf.debugging.set_log_device_placement(True)
            configure_threads(1, 1)
            configure_gpu()

            self.env_name = env_config['name']
            env = create_gym_env(env_config)
            data_format = dict(
                state=(env.state_dtype, (None, *env.state_shape)),
                action=(env.action_dtype, (None, *env.action_shape)),
                reward=(tf.float32, (None, )), 
                next_state=(env.state_dtype, (None, *env.state_shape)),
                done=(tf.float32, (None, )),
                steps=(tf.float32, (None, )),
            )
            dataset = RayDataset(replay, data_format)
            self.model = Ensemble(model_fn, model_config, env.state_shape, env.action_dim, env.is_action_discrete)
            
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
            self.writer.set_as_default()
            while True:
                step += 1
                with TBTimer(f'{self.name} train', 10000, to_log=self.timer):
                    self.learn_log(step)
                if self.weight_repo and step % 1000 == 0:
                    print_repo(self.weight_repo, self.model_name, self.cb_c)
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
            pop_score = store_weights(
                self.weight_repo, mode, score, tag, weights, 
                eval_times, self.REPO_CAP, fifo=self.FIFO,
                fitness_method=self.fitness_method, c=self.cb_c)

            self._make_decision(score, tag, eval_times, pop_score)

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
                w = fitness_from_repo(self.weight_repo, 'ucb', c=self.cb_c)
                score = random.choices(list(self.weight_repo), weights=w)[0]
                tag = self.weight_repo[score].tag
                weights = self.weight_repo[score].weights
                eval_times = self.weight_repo[score].eval_times
                del self.weight_repo[score]
            else:
                raise ValueError(f'Unknown mode: {mode}')
        
            return mode, score, tag, weights, eval_times
        
        def _make_decision(self, score, tag, eval_times, pop_score):
            if score != pop_score:
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

    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    config['model_name'] = 'learner'
    config['mode_prob'] = [1, 0, 0]
    # learner only define a env to get necessary env info, 
    # it does not actually interact with env
    env_config['n_workers'] = env_config['n_envs'] = 1

    if env_config.get('is_deepmind_env'):
        RayLearner = ray.remote(num_cpus=2, num_gpus=.5)(Learner)
    else:
        if tf.config.list_physical_devices('GPU'):
            RayLearner = ray.remote(num_cpus=1, num_gpus=.1)(Learner)
        else:
            RayLearner = ray.remote(num_cpus=1)(Learner)
    learner = RayLearner.remote(name, model_fn, replay, config, 
                            model_config, env_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner
