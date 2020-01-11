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
    @ray.remote(num_cpus=1, num_gpus=.1)
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
            dataset = RayDataset(replay, env.state_shape, env.state_dtype, env.action_shape, env.action_dtype, env.action_dim)
            self.model = Ensemble(model_fn, model_config, env.state_shape, env.action_dim, env.is_action_discrete)
            
            super().__init__(
                name=name, 
                config=config, 
                models=self.model,
                dataset=dataset,
                env=env,
            )

            self.weight_repo = {}                                 # map score to Weights
            self.mode = (Mode.LEARNING, Mode.EVOLUTION, Mode.REEVALUATION)
            self.raw_bookkeeping = BookKeeping('raw')
            self.info_to_print = []
            self.records = {}
            
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
                if step % 1000 == 0:
                    print_repo(self.weight_repo, self.model_name)
                    self.store(mode_learned=self.mode_prob[0], model_evolved=self.mode_prob[1])
                    self.store(**analyze_repo(self.weight_repo))
                    self.log(step, print_terminal_info=False)
                if step % 100000 == 0:
                    self.save(print_terminal_info=False)

        def get_weights(self, worker_id, name=None):
            mode, score, tag, weights, eval_times = self._choose_weights(worker_id, name=name)
            
            self.records[worker_id] = Weights(tag, weights, eval_times)

            threshold = min(self.weight_repo) if self.weight_repo else -float('inf')

            return threshold, mode, score, tag, weights, eval_times

        def store_weights(self, score, tag, weights, eval_times):
            store_weights(self.weight_repo, score, tag, weights, eval_times, self.REPO_CAP, fifo=self.FIFO)

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
                weights, n = evolve_weights(self.weight_repo, min_evolv_models=self.MIN_EVOLVE_MODELS, 
                            max_evolv_models=self.MAX_EVOLVE_MODELS, weighted_average=self.EVOLVE_WA)
                self.info_to_print.append(((f'{self.name}: {n} models are used for evolution', ), 'blue'))
                score = 0
                eval_times = 0
            elif mode == Mode.REEVALUATION:
                score = random.choices(list(self.weight_repo.keys()))[0]
                tag = self.weight_repo[score].tag
                weights = self.weight_repo[score].weights
                eval_times = self.weight_repo[score].eval_times
                del self.weight_repo[score]
            else:
                raise ValueError(f'Unknown mode: {mode}')
        
            return mode, score, tag, weights, eval_times
 
        def _update_mode_prob(self):
            fracs = analyze_repo(self.weight_repo)
            if self.env_name == 'BipedalWalkerHardcore-v2' and min(self.weight_repo) > 300:
                self.mode_prob[2] = 1
                self.mode_prob[0] = self.mode_prob[1] = 0
                return
            else:
                self.mode_prob[2] = self.EVAL_PROB
            remain_prob = 1 - self.mode_prob[2] - self.MIN_LEARN_PROB - self.MIN_EVOLVE_PROB

            self.mode_prob[0] = self.MIN_LEARN_PROB + fracs['frac_learned'] * remain_prob
            self.mode_prob[1] = self.MIN_EVOLVE_PROB + fracs['frac_evolved'] * remain_prob
            np.testing.assert_allclose(sum(self.mode_prob), 1)

    config = config.copy()
    model_config = model_config.copy()
    env_config = env_config.copy()
    replay_config = replay_config.copy()
    
    config['model_name'] = 'learner'
    config['mode_prob'] = [1, 0, 0]
    # learner only define a env to get necessary env info, 
    # it does not actually interact with env
    env_config['n_workers'] = env_config['n_envs'] = 1

    learner = Learner.remote(name, model_fn, replay, config, 
                            model_config, env_config)
    ray.get(learner.save_config.remote(dict(
        env=env_config,
        model=model_config,
        agent=config,
        replay=replay_config
    )))

    return learner
