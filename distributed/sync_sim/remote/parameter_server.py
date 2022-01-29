import collections
import os
import random
from typing import Dict, List
import cloudpickle
import ray
from ray.util.queue import Queue

from .typing import ModelWeights
from .utils import get_aid, get_aid_vid
from core.elements.builder import ElementsBuilderVC
from core.mixin.actor import RMSStats, combine_rms_stats
from core.remote.base import RayBase
from core.typing import ModelPath
from run.utils import search_for_config
from utility.utils import dict2AttrDict


payoff = collections.defaultdict(lambda: collections.deque(maxlen=1000))
score = collections.defaultdict(lambda: 0)


class ParameterServer(RayBase):
    def __init__(
        self, 
        configs: dict, 
        env_stats: dict,
        param_queues: List[List[Queue]],
        name='parameter_server'
    ):
        super().__init__()
        self.configs = configs
        self.param_queues = param_queues
        self.name = name

        configs = [dict2AttrDict(config['parameter_server']) for config in configs]
        self.config = configs[0]
        self.n_agents = len(configs)

        self.builders: List[ElementsBuilderVC] = []
        for aid, config in enumerate(self.configs):
            path = f'{config["root_dir"]}/{config["model_name"]}'
            assert path.rsplit('/')[-1] == f'a{aid}', path
            os.makedirs(path, exist_ok=True)
            builder = ElementsBuilderVC(config, env_stats, to_save_code=False)
            self.builders.append(builder)
        self._ps_dir = path.rsplit('/', 1)[0]
        os.makedirs(self._ps_dir, exist_ok=True)
        self._path = f'{self._ps_dir}/path.pkl'

        self._params: List[Dict[ModelPath, Dict]] = [{} for _ in configs]
        # self._payoffs: Dict[Tuple[ModelPath], Deque[float]] = payoff
        # self._scores: Dict[Tuple[ModelPath], float] = collections.defaultdict(lambda: 0)

        # an active model is the one under training
        self._active_model_paths = [None for _ in configs]
        self._train_from_scratch_frac = self.config.get('train_from_scratch_frac', 1)


        self.restore()

    def get_configs(self):
        return self.configs

    """ Strategy Management """
    def update_strategy_weights(self, aid, model_weights: ModelWeights):
        assert self._active_model_paths[aid] == model_weights.model, \
            (self._active_model_paths, model_weights.model)
        assert set(model_weights.weights) == set(['model', 'opt', 'train_step']), \
            list(model_weights.weights)
        model, weights = model_weights
        self._params[aid][model].update(weights)

        # put the latest parameters in the queue for runners to retrieve
        weights.pop('opt')
        weights['aux'] = self._params[aid][model].get('aux', RMSStats(None, None))
        mid = ray.put(model_weights)
        for q in self.param_queues[aid]:
            q.put(mid)

    def update_strategy_aux_stats(self, aid, model_weights: ModelWeights):
        assert len(model_weights.weights) == 1, list(model_weights.weights)
        assert 'aux' in model_weights.weights, list(model_weights.weights)
        if self._params[aid][model_weights.model] is not None \
                and 'aux' in self._params[aid][model_weights.model]:
            self._params[aid][model_weights.model]['aux'] = combine_rms_stats(
                self._params[aid][model_weights.model]['aux'], 
                model_weights.weights['aux'],
            )
        else:
            self._params[aid][model_weights.model]['aux'] = model_weights.weights['aux']

    def sample_training_strategies(self):
        strategies = []
        is_raw_strategy = [False for _ in range(self.n_agents)]
        if any([am is not None for am in self._active_model_paths]):
            for aid, path in enumerate(self._active_model_paths):
                weights = self._params[aid][path].copy()
                weights.pop('aux', None)
                strategies.append(ModelWeights(path, weights))
        else:
            assert all([am is None for am in self._active_model_paths]), self._active_model_paths
            for aid in range(self.n_agents):
                if random.random() < self._train_from_scratch_frac:
                    self.builders[aid].increase_version()
                    path = self.builders[aid].get_model_path()
                    self._params[aid][path] = {}
                    weights = None
                    is_raw_strategy[aid] = True
                else:
                    path = random.choice(list(self._params[aid]))
                    weights = self._params[aid][path].copy()
                    weights.pop('aux')
                    config = search_for_config(path)
                    path, config = self.builders[aid].get_sub_version(config)
                    print('sample_training_strategies: version', path, config.version)
                self._active_model_paths[aid] = path
                strategies.append(ModelWeights(path, weights))
            self.save()

        return strategies, is_raw_strategy

    def archive_training_strategies(self):
        for model_path in self._active_model_paths:
            self.save_params(model_path)
        self._active_model_paths = [None for _ in range(self.n_agents)]
        self.save()

    """ Checkpoints """
    def save_active_model(self, model, train_step, env_step):
        is_model_active = False
        for aid, model_path in enumerate(self._active_model_paths):
            if model == model_path:
                is_model_active = True
                self._params[aid][model]['train_step'] = train_step
                self._params[aid][model]['env_step'] = env_step
                self.save_params(model)
                break
        assert is_model_active, f'{model} is not active!'

    def save_params(self, model: ModelPath):
        assert model in self._active_model_paths, (model, self._active_model_paths)
        assert self._ps_dir.rsplit('/', 1)[-1] == model.model_name.split('/', 1)[0], (
            self._ps_dir, model.model_name
        )
        aid, vid = get_aid_vid(model.model_name)
        ps_dir = f'{self._ps_dir}/a{aid}'
        if not os.path.isdir(ps_dir):
            os.makedirs(ps_dir)
        path = f'{ps_dir}/v{vid}/params.pkl'
        with open(path, 'wb') as f:
            cloudpickle.dump(self._params[aid][model], f)
        print(f'Save parameters in "{path}"')

    def restore_params(self, model: ModelPath):
        aid, vid = get_aid_vid(model.model_name)
        path = f'{self._ps_dir}/a{aid}/v{vid}/params.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._params[aid][model] = cloudpickle.load(f)
            print(f'Restore parameters from "{path}"')
        else:
            self._params[aid][model] = {}

    def save(self):
        with open(self._path, 'wb') as f:
            cloudpickle.dump(
                ([list(p) for p in self._params], self._active_model_paths), f)

    def restore(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, 'rb') as f:
                    model_paths, self._active_model_paths = cloudpickle.load(f)
            except Exception as e:
                print(f'Error happened when restoring from {self._path}: {e}')
                return
            for models in model_paths:
                for m in models:
                    self.restore_params(m)

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

        aid = get_aid(model_path.model_name)
        rms = self._params[aid][model_path]['aux']
        stats = rms2dict(rms)

        return stats


if __name__ == '__main__':
    from env.func import get_env_stats
    from utility.yaml_op import load_config
    config = load_config('algo/gd/configs/builtin.yaml')
    env_stats = get_env_stats(config['env'])
    ps = ParameterServer(config, env_stats)
