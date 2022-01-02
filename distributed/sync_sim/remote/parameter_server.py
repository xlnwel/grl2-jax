import collections
import os
import random
import time
from typing import Deque, Dict, List, Tuple
import cloudpickle
import ray
from ray.util.queue import Queue

from .typing import ModelWeights
from .utils import get_aid_vid
from core.elements.builder import ElementsBuilder
from core.elements.strategy import Strategy
from core.remote.base import RayBase
from core.typing import ModelPath
from env.func import get_env_stats
from run.utils import search_for_all_configs, search_for_config
from utility.display import pwc
from utility.utils import dict2AttrDict


payoff = collections.defaultdict(lambda: collections.deque(maxlen=1000))
score = collections.defaultdict(lambda: 0)


class ParameterServer(RayBase):
    def __init__(self, 
                 configs: dict, 
                 param_queues: List[List[Queue]],
                 name='parameter_server'):
        super().__init__()
        self.param_queues = param_queues
        self.builder = ElementsBuilder(configs[0])

        configs = [dict2AttrDict(config['parameter_server']) for config in configs]
        self.config = configs[0]
        self.n_agents = len(configs)

        for config in configs:
            path = f'{config.root_dir}/{config.model_name}'
            if not os.path.exists(path):
                os.makedirs(path)
        self._stats_path = f'{path}/{name}.pkl'
        
        self._strategies: List[Dict[ModelPath, Strategy]] = [{} for _ in configs]
        self._payoffs: Dict[Tuple[ModelPath], Deque[float]] = payoff
        self._scores: Dict[Tuple[ModelPath], float] = collections.defaultdict(lambda: 0)

        self._active_strategies_path = [None for _ in configs]
        self._start_checkpoints = [
            config.start_checkpoint and ModelPath(*config.start_checkpoint) 
            for config in configs]

        self.restore()
        self._prev_save_time = time.time()

    def search_for_strategies(self, strategy_dir):
        pwc(f'Parameter server: searching for strategies in directory({strategy_dir})', color='cyan')
        configs = search_for_all_configs(strategy_dir)
        for c in configs:
            self.add_strategy_from_config(c)
    
    """ Add Strategies """
    def add_strategy_from_config(self, config, set_active=False):
        assert not config.model_name.startswith('/'), config.model_name
            # config.model_name = config.model_name[1:]
            # save_config(config)
        aid, _ = get_aid_vid(config.model_name)
        model_path = ModelPath(config.root_dir, config.model_name)
        if set_active:
            self._active_strategies_path[aid] = model_path
        if model_path not in self._strategies[aid]:
            config.trainer.display_var = False
            elements = self.builder.build_strategy_from_scratch(
                config, build_monitor=False, save_config=False)
            elements.strategy.restore()
            self._strategies[aid][model_path] = elements.strategy
            self.save()
        print(f'Strategies for Agent({aid}) in the pool:', list(self._strategies[aid]))

    def add_strategy_from_path(self, aid, model_path: ModelPath, set_active=False):
        if model_path not in self._strategies[aid]:
            path = '/'.join(model_path)
            config = search_for_config(path)
            self.add_strategy_from_config(config, set_active=set_active)
        if set_active:
            self._active_strategies_path[aid] = model_path

    """ Update Strategies """
    def update_strategy_weights(self, aid, model_weights: ModelWeights):
        assert self._active_strategies_path[aid] == model_weights.model, \
            (self._active_strategies_path, model_weights.model)
        assert len(model_weights.weights) == 3, list(model_weights.weights)
        model, weights = model_weights
        if model not in self._strategies[aid]:
            self.add_strategy_from_path(aid, model)
        self._strategies[aid][model].set_weights(weights)

        # put the latest parameters in the queue for runners to retrieve
        model_weights.weights.pop('opt')
        model_weights.weights['aux'] = self._strategies[aid][model].actor.get_auxiliary_stats()
        mid = ray.put(model_weights)
        for q in self.param_queues[aid]:
            q.put(mid)
    
    def update_strategy_aux_stats(self, aid, model_weights: ModelWeights):
        assert len(model_weights.weights) == 1, list(model_weights.weights)
        assert 'aux' in model_weights.weights, list(model_weights.weights)
        self._strategies[aid][model_weights.model].actor.update_rms_from_stats(model_weights.weights['aux'])

    """ Sample Strategies"""
    def sample_strategy_path(self, aid, model_path: ModelPath):
        aid2, _ = get_aid_vid(model_path.model_name)
        assert aid == aid2, (aid, aid2)
        scores = self.get_scores(model_path.model_name)
        weights = self.get_weights_vector(scores)
        return random.choices([k for k in self._strategies[aid] if k != model_path], weights=weights)[0]

    def sample_strategy(self, aid, model_path: ModelPath):
        path = self.sample_strategy_path(aid, model_path)
        strategy = self._strategies[aid][path]
        weights = strategy.get_weights()

        return path, ModelWeights(path, weights)

    def retrieve_start_strategies_weights(self):
        if self._start_checkpoints:
            model_weights = []
            for aid, path in enumerate(self._start_checkpoints):
                strategy = self._strategies[aid][path]
                weights = strategy.get_weights()
                model_weights.append(ModelWeights(path, weights))
            return model_weights
        else:
            return None

    def retrieve_active_strategies_weights(self):
        if self._active_strategies_path:
            model_weights = []
            for aid, path in enumerate(self._active_strategies_path):
                strategy = self._strategies[aid][path]
                weights = strategy.get_weights()
                model_weights.append(ModelWeights(path, weights))
            return model_weights
        else:
            return None

    def retrieve_start_strategies_paths(self):
        return self._start_checkpoints

    def retrieve_latest_strategies_paths(self):
        return self._active_strategies_path

    """ Checkpoints """
    def save_active_models(self, model, train_step, env_step):
        is_model_active = False
        for aid, model_path in enumerate(self._active_strategies_path):
            if model == model_path:
                is_model_active = True
                self._strategies[aid][model_path].set_train_step(train_step)
                self._strategies[aid][model_path].set_env_step(env_step)
                self._strategies[aid][model_path].save(print_terminal_info=True)
                self.save()
                break
        assert is_model_active, f'{model} is not active!'

    def save(self):
        with open(self._stats_path, 'wb') as f:
            cloudpickle.dump(
                ([list(s) for s in self._strategies], 
                self._payoffs, self._scores), f)

    def restore(self):
        if os.path.exists(self._stats_path):
            try:
                with open(self._stats_path, 'rb') as f:
                    paths, self._payoffs, self._scores = cloudpickle.load(f)
            except Exception as e:
                print(f'Error happened when restoring from {self._stats_path}: {e}')
                return
            for aid, paths in enumerate(paths):
                for p in paths:
                    try:
                        self.add_strategy_from_path(aid, p)
                    except Exception as e:
                        print(f'Skip {p} as it is no longer a valid path: {e}')

    """ Data Retrieval """
    def get_aux_stats(self, model_path: ModelPath):
        def rms2dict(rms):
            d = {}
            for k, v in rms.obs.items():
                for kk, vv in v._asdict().items():
                    d[f'aux/{k}/{kk}'] = vv
            for k, v in rms.reward._asdict().items():
                d[f'aux/reward/{k}'] = v

            return d

        aid, _ = get_aid_vid(model_path.model_name)
        rms = self._strategies[aid][model_path].actor.get_auxiliary_stats()
        stats = rms2dict(rms)

        return stats


if __name__ == '__main__':
    from env.func import get_env_stats
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env_stats = get_env_stats(config['env'])
    ps = ParameterServer(config, env_stats)
