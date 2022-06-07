import os
import logging
from types import FunctionType
from typing import Dict, Tuple
import cloudpickle

from core.elements.actor import Actor
from core.elements.dataset import create_dataset
from core.elements.model import Model
from core.elements.strategy import Strategy
from core.elements.trainer import Trainer
from core.log import do_logging
from core.monitor import Monitor, create_monitor
from core.typing import ModelPath
from core.utils import save_code, save_config
from env.func import get_env_stats
from utility.display import pwt
from utility.utils import AttrDict2dict, dict2AttrDict, set_path
from utility.typing import AttrDict
from utility import pkg


logger = logging.getLogger(__name__)


class ElementsBuilder:
    def __init__(
        self, 
        config: dict, 
        env_stats: dict=None, 
        to_save_code: bool=False,
        name='builder'
    ):
        self.config = dict2AttrDict(config, to_copy=True)
        self.env_stats = dict2AttrDict(
            get_env_stats(self.config.env) if env_stats is None else env_stats)
        self._name = name

        self._model_path = ModelPath(self.config.root_dir, self.config.model_name)

        algo = self.config.algorithm.split('-')[-1]
        self.constructors = self.retrieve_constructor(algo)

        if to_save_code:
            save_code(self._model_path)
            pwt('Save code', self._model_path)

    @property
    def name(self):
        return self._name
    
    def retrieve_constructor(self, algo):
        constructors = AttrDict()
        constructors.model = self._import_element(
            name='model', algo=algo, place=-1).create_model
        constructors.loss = self._import_element(
            name='loss', algo=algo, place=-1).create_loss
        constructors.trainer = self._import_element(
            name='trainer', algo=algo, place=-1).create_trainer
        constructors.actor = self._import_element(
            name='actor', algo=algo, place=-1).create_actor
        constructors.buffer = self._import_element(
            'buffer', algo=algo).create_buffer
        constructors.strategy = self._import_element(
            'strategy', algo=algo).create_strategy
        constructors.agent = self._import_element(
            'agent', algo=algo).create_agent

        return constructors

    def get_model_path(self):
        return self._model_path

    """ Build Elements """
    def build_model(
        self, 
        config: dict=None, 
        env_stats: dict=None, 
        to_build: bool=False, 
        to_build_for_eval: bool=False, 
        constructors: Dict[str, FunctionType]=None
    ):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        model = constructors.model(
            config.model, env_stats, name=config.name,
            to_build=to_build, to_build_for_eval=to_build_for_eval)
        
        return model

    def build_actor(
        self, 
        model: Model, 
        config: dict=None, 
        constructors: Dict[str, FunctionType]=None
    ):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        actor = constructors.actor(config.actor, model, name=config.name)
        
        return actor
    
    def build_trainer(
        self, 
        model: Model, 
        config: dict=None, 
        env_stats: dict=None, 
        constructors: Dict[str, FunctionType]=None
    ):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        loss = constructors.loss(config.loss, model, name=config.name)
        trainer = constructors.trainer(
            config.trainer, env_stats, loss, name=config.name)
        
        return trainer
    
    def build_buffer(
        self, 
        model: Model, 
        config: dict=None, 
        env_stats: dict=None, 
        constructors: Dict[str, FunctionType]=None, 
        **kwargs
    ):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        buffer = constructors.buffer(
            config.buffer, 
            model, 
            env_stats, 
            **kwargs
        )
        
        return buffer

    def build_dataset(
        self, 
        buffer, 
        model: Model, 
        config: dict=None, 
        env_stats: dict=None, 
        central_buffer: bool=False
    ):
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        if self.config.buffer['use_dataset']:
            am = pkg.import_module(
                'elements.utils', algo=config.algorithm, place=-1)
            data_format = am.get_data_format(
                self.config.trainer, env_stats, model)
            dataset = create_dataset(buffer, env_stats, 
                data_format=data_format, central_buffer=central_buffer, 
                one_hot_action=False)
        else:
            dataset = buffer

        return dataset
    
    def build_strategy(
        self, 
        actor: Actor=None, 
        trainer: Trainer=None, 
        dataset=None, 
        config: dict=None, 
        env_stats: dict=None, 
        constructors: Dict[str, FunctionType]=None
    ):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        strategy = constructors.strategy(
            config.name, 
            config.strategy, 
            env_stats=env_stats,
            actor=actor, 
            trainer=trainer, 
            dataset=dataset)
        
        return strategy

    def build_monitor(
        self, 
        config: dict=None, 
        save_to_disk: bool=True
    ):
        if save_to_disk:
            config = dict2AttrDict(config or self.config)
            return create_monitor(ModelPath(config.root_dir, config.model_name))
        else:
            return create_monitor(None, use_tensorboard=False)
    
    def build_agent(
        self, 
        strategy: Strategy, 
        monitor: Monitor=None, 
        config: dict=None, 
        constructors: FunctionType=None,
        to_restore: bool=True
    ):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        agent = constructors.agent(
            config=config.agent, 
            strategy=strategy, 
            monitor=monitor, 
            name=config.name, 
            to_restore=to_restore
        )

        return agent

    """ Build an Strategy/Agent from Scratch .
    We delibrately define different interfaces for each type of 
    Strategy/Agent to offer default setups for a variety of situations
    
    An acting strategy/agent is used to interact with environments only
    A training strategy/agent is used for training only
    A plain strategy/agent is used for both cases
    """
    def build_acting_strategy_from_scratch(
        self, 
        config: dict=None, 
        env_stats: dict=None,
        build_monitor: bool=True, 
        to_build_for_eval: bool=False
    ):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if env_stats is None else env_stats
        
        elements = AttrDict()
        elements.model = self.build_model(
            config=config, 
            env_stats=env_stats, 
            to_build=not to_build_for_eval, 
            to_build_for_eval=to_build_for_eval,
            constructors=constructors)
        elements.actor = self.build_actor(
            model=elements.model, 
            config=config,
            constructors=constructors)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            config=config,
            env_stats=env_stats,
            constructors=constructors)
        if build_monitor:
            elements.monitor = self.build_monitor(
                config=config, 
                save_to_disk=False)

        return elements
    
    def build_training_strategy_from_scratch(
        self, 
        config: dict=None, 
        env_stats: dict=None,
        build_monitor: bool=True, 
        save_monitor_stats_to_disk: bool=False,
        save_config: bool=True
    ):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if env_stats is None else env_stats
        
        elements = AttrDict()
        elements.model = self.build_model(
            config=config, 
            env_stats=env_stats, 
            constructors=constructors)
        elements.trainer = self.build_trainer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.buffer = self.build_buffer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats, 
            constructors=constructors)
        elements.dataset = self.build_dataset(
            buffer=elements.buffer, 
            model=elements.model, 
            config=config,
            env_stats=env_stats)
        elements.strategy = self.build_strategy(
            trainer=elements.trainer, 
            dataset=elements.dataset, 
            config=config,
            env_stats=env_stats,
            constructors=constructors)
        if build_monitor:
            elements.monitor = self.build_monitor(
                config=config, 
                save_to_disk=save_monitor_stats_to_disk)

        if save_config:
            self.save_config()

        return elements
    
    def build_strategy_from_scratch(
        self, 
        config: dict=None, 
        env_stats: dict=None,
        build_monitor: bool=True, 
        save_monitor_stats_to_disk: bool=False,
        save_config: bool=True
    ):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if env_stats is None else env_stats
        
        elements = AttrDict()
        elements.model = self.build_model(
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.actor = self.build_actor(
            model=elements.model, 
            config=config,
            constructors=constructors)
        elements.trainer = self.build_trainer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.buffer = self.build_buffer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats, 
            constructors=constructors)
        elements.dataset = self.build_dataset(
            buffer=elements.buffer, 
            model=elements.model, 
            config=config,
            env_stats=env_stats)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            trainer=elements.trainer, 
            dataset=elements.dataset,
            config=config,
            env_stats=env_stats,
            constructors=constructors)
        if build_monitor:
            elements.monitor = self.build_monitor(
                config=config, 
                save_to_disk=save_monitor_stats_to_disk)

        if save_config:
            self.save_config()

        return elements

    def build_acting_agent_from_scratch(
        self, 
        config: dict=None, 
        env_stats: dict=None,
        build_monitor: bool=True, 
        to_build_for_eval: bool=False,
        to_restore: bool=True
    ):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if env_stats is None else env_stats
        
        elements = AttrDict()
        elements.model = self.build_model(
            config=config, 
            env_stats=env_stats,
            to_build=not to_build_for_eval, 
            to_build_for_eval=to_build_for_eval,
            constructors=constructors)
        elements.actor = self.build_actor(
            model=elements.model, 
            config=config,
            constructors=constructors)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            config=config,
            env_stats=env_stats,
            constructors=constructors)
        if build_monitor:
            elements.monitor = self.build_monitor(
                config=config, 
                save_to_disk=False)
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor if build_monitor else None, 
            config=config,
            constructors=constructors,
            to_restore=to_restore)

        return elements
    
    def build_training_agent_from_scratch(
        self, 
        config: dict=None, 
        env_stats: dict=None,
        save_monitor_stats_to_disk: bool=True,
        save_config: bool=True
    ):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if env_stats is None else env_stats
        
        elements = AttrDict()
        elements.model = self.build_model(
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.trainer = self.build_trainer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.buffer = self.build_buffer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats, 
            constructors=constructors)
        elements.dataset = self.build_dataset(
            buffer=elements.buffer, 
            model=elements.model, 
            config=config,
            env_stats=env_stats)
        elements.strategy = self.build_strategy(
            trainer=elements.trainer, 
            dataset=elements.dataset,
            config=config,
            env_stats=env_stats,
            constructors=constructors)
        elements.monitor = self.build_monitor(
            config=config,
            save_to_disk=save_monitor_stats_to_disk)
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor,
            config=config,
            constructors=constructors)
        
        if save_config:
            self.save_config()

        return elements

    def build_agent_from_scratch(
        self, 
        config: dict=None, 
        env_stats: dict=None,
        save_monitor_stats_to_disk: bool=True,
        save_config: bool=True
    ):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if env_stats is None else env_stats
        
        elements = AttrDict()
        elements.model = self.build_model(
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.actor = self.build_actor(
            model=elements.model, 
            config=config,
            constructors=constructors)
        elements.trainer = self.build_trainer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats,
            constructors=constructors)
        elements.buffer = self.build_buffer(
            model=elements.model, 
            config=config, 
            env_stats=env_stats, 
            constructors=constructors)
        elements.dataset = self.build_dataset(
            buffer=elements.buffer, 
            model=elements.model, 
            config=config,
            env_stats=env_stats)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            trainer=elements.trainer, 
            dataset=elements.dataset,
            config=config,
            env_stats=env_stats,
            constructors=constructors)
        elements.monitor = self.build_monitor(
            config=config,
            save_to_disk=save_monitor_stats_to_disk)
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor,
            config=config,
            constructors=constructors)
        
        if save_config:
            self.save_config()

        return elements

    def get_constructors(self, config: AttrDict):
        if config is not None and config.algorithm != self.config.algorithm:
            constructors = self.retrieve_constructor(config.algorithm)
        else:
            constructors = self.constructors
        return constructors

    """ Configuration Operations """
    def get_config(self):
        return self.config

    def save_config(self, config: dict=None):
        save_config(config or self.config)
        pwt('Save config', ModelPath(self.config.root_dir, self.config.model_name))

    """ Implementations"""
    def _import_element(self, name, algo=None, *, config=None, place=0):
        try:
            module = pkg.import_module(
                f'elements.{name}', algo=algo)
        except Exception as e:
            level = 'info' if name == 'agent' else 'pwc'
            do_logging(
                f'Switch to default module({name}) due to error: {e}', 
                logger=logger, level=level)
            do_logging(
                "You are safe to neglect it if it's an intended behavior. ", 
                logger=logger, level=level)
            module = pkg.import_module(
                f'elements.{name}', pkg='core')
        return module


""" Element Builder with Version Control """
class ElementsBuilderVC(ElementsBuilder):
    def __init__(
        self, 
        config: dict, 
        env_stats: dict=None, 
        start_version=0, 
        to_save_code=False, 
        name='builder', 
    ):
        super().__init__(
            config, 
            env_stats=env_stats, 
            to_save_code=to_save_code, 
            name=name
        )

        self._default_model_path = self._model_path
        self._builder_path = '/'.join([*self._model_path, f'{self._name}.pkl'])

        self._version: str = str(start_version)
        self._all_versions = set()
        self._max_version = start_version
        self.restore()

    """ Version Control """
    def get_version(self):
        return self._version

    def increase_version(self):
        self._max_version += 1
        self.set_config_version(self._max_version)
        self._all_versions.add(self._max_version)
        self.save_config()
        self.save()

    def get_sub_version(self, config: AttrDict) -> Tuple[ModelPath, AttrDict]:
        def compute_next_version(version: str):
            def next_version(base_version: str, sub_version: str):
                sub_version = eval(sub_version)
                new_sub_version = f'{sub_version + 1}'
                version = '.'.join([base_version, new_sub_version])
                return version

            if '.' in version:
                base_version, sub_version = version.rsplit('.', maxsplit=1)
            else:
                base_version = version
                sub_version = '0'
            version = next_version(base_version, sub_version)
            if version in self._all_versions:
                base_version = '.'.join([base_version, sub_version])
                sub_version = '0'
            version = next_version(base_version, sub_version)

            return version

        root_dir = config.root_dir
        model_name = config.model_name
        model_name, version = model_name.rsplit('/', maxsplit=1)
        assert version.startswith('v'), (model_name, version)
        version = version[1:]   # remove prefix v
        assert version == f'{config.version}', (version, config.version)
        version = compute_next_version(version)
        model_name = f'{model_name}/v{version}'
        model_path = ModelPath(root_dir, model_name)
        model_dir = '/'.join(model_path)
        config = set_path(config, model_path)
        config.version = version
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        self.save_config(config)
        self._all_versions.add(self._max_version)

        return model_path, config

    def set_config_version(self, version: float):
        root_dir = self.config.root_dir
        model_name = f'{self._default_model_path.model_name}/v{version}'
        self._model_path = ModelPath(root_dir, model_name)
        self.config = set_path(self.config, self._model_path)
        self._version = version
        self.config.version = str(version)
        self._model_path, self.config = self.get_sub_version(self.config)

    """ Save & Restore """
    def restore(self):
        if os.path.exists(self._builder_path):
            with open(self._builder_path, 'rb') as f:
                self._version, self._max_version, config = cloudpickle.load(f)
                # we only restore the model_path but not other configs
                model = ModelPath(config['root_dir'], config['model_name'])
                self.config = set_path(self.config, model)
                # self.config = dict2AttrDict(config)

    def save(self):
        with open(self._builder_path, 'wb') as f:
            cloudpickle.dump((self._version, self._max_version, AttrDict2dict(self.config)), f)

if __name__ == '__main__':
    from env.func import create_env
    from utility.yaml_op import load_config
    config = load_config('algo/zero/configs/card.yaml')
    env = create_env(config['env'])
    config.model_name = 'test'
    env_stats = env.stats()
    builder = ElementsBuilder(config, env_stats=env_stats, name='zero', incremental_version=True)
    elements = builder.build_agent_from_scratch()
    elements.agent.save()
    builder.increase_version()
    elements = builder.build_agent_from_scratch()
    elements.agent.save()
