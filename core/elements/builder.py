import os
import cloudpickle

from core.elements.dataset import create_dataset
from core.monitor import create_monitor
from core.typing import ModelPath
from core.utils import save_code, save_config
from env.func import get_env_stats
from utility.utils import set_path
from utility.typing import AttrDict
from utility.utils import dict2AttrDict
from utility import pkg


class ElementsBuilder:
    def __init__(self, 
                 config, 
                 env_stats=None, 
                 incremental_version=False,
                 start_version=0,
                 to_save_code=False):
        self.config = dict2AttrDict(config, to_copy=True)
        self.env_stats = dict2AttrDict(
            get_env_stats(self.config.env) if env_stats is None else env_stats)
        self._name = self.config.name

        self._model_path = ModelPath(self.config.root_dir, self.config.model_name)
        self._default_model_path = self._model_path
        self._builder_path = '/'.join([*self._model_path, f'{self._name}_builder.pkl'])

        self._incremental_version = incremental_version
        self._version = start_version
        self._max_version = start_version
        if self._incremental_version:
            self.restore()
            self.set_config_version(self._version)

        algo = self.config.algorithm.split('-')[-1]
        self.constructors = self.retrieve_constructor(algo)

        if to_save_code:
            save_code(self._default_model_path)

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

    def get_version(self):
        return self._version

    def increase_version(self):
        self._max_version += 1
        self.set_config_version(self._max_version)
        self.save_config()
        self.save()

    def set_config_version(self, version):
        if self._incremental_version:
            root_dir = self.config.root_dir
            model_name = f'{self._default_model_path.model_name}/v{version}'
            self._model_path = ModelPath(root_dir, model_name)
            model_dir = '/'.join(self._model_path)
            self.config = set_path(self.config, self._model_path)
            self.config.version = self._version = version
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir, exist_ok=True)

    def get_model_path(self):
        return self._model_path

    """ Build Elements """
    def build_model(self, config=None, env_stats=None, to_build=False, 
            to_build_for_eval=False, constructors=None):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        model = constructors.model(
            config.model, env_stats, name=config.name,
            to_build=to_build, to_build_for_eval=to_build_for_eval)
        
        return model

    def build_actor(self, model, config=None, constructors=None):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        actor = constructors.actor(config.actor, model, name=config.name)
        
        return actor
    
    def build_trainer(self, model, config=None, env_stats=None, constructors=None):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        loss = constructors.loss(config.loss, model, name=config.name)
        trainer = constructors.trainer(
            config.trainer, env_stats, loss, name=config.name)
        
        return trainer
    
    def build_buffer(self, model, config=None, constructors=None, **kwargs):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        buffer = constructors.buffer(config.buffer, model, **kwargs)
        
        return buffer

    def build_dataset(self, buffer, model, config=None, env_stats=None, central_buffer=False):
        config = dict2AttrDict(config or self.config)
        env_stats = dict2AttrDict(env_stats or self.env_stats)
        if self.config.buffer['use_dataset']:
            am = pkg.import_module('elements.utils', algo=config.algorithm)
            data_format = am.get_data_format(
                self.config.trainer, env_stats, model)
            dataset = create_dataset(buffer, env_stats, 
                data_format=data_format, central_buffer=central_buffer, 
                one_hot_action=False)
        else:
            dataset = buffer

        return dataset
    
    def build_strategy(self, actor=None, trainer=None, dataset=None, config=None, env_stats=None, constructors=None):
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

    def build_monitor(self, config=None, save_to_disk=True):
        if save_to_disk:
            config = dict2AttrDict(config or self.config)
            return create_monitor(ModelPath(config.root_dir, config.model_name), self.name)
        else:
            return create_monitor(None, self.name, use_tensorboard=False)
    
    def build_agent(self, strategy, monitor=None, config=None, constructors=None):
        constructors = constructors or self.constructors
        config = dict2AttrDict(config or self.config)
        agent = constructors.agent(
            config=config.agent, 
            strategy=strategy, 
            monitor=monitor, 
            name=config.name, 
        )

        return agent

    """ Build an Strategy/Agent from Scratch .
    We delibrately define different interfaces for each type of 
    Strategy/Agent to offer default setups for a variety of situations
    
    An acting strategy/agent is used to interact with environments only
    A training strategy/agent is used for training only
    A plain strategy/agent is used for both cases
    """
    def build_acting_strategy_from_scratch(self, 
            config=None, 
            build_monitor=True, 
            to_build_for_eval=False):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if config is None else get_env_stats(config.env)
        
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
    
    def build_training_strategy_from_scratch(self, 
            config=None, 
            build_monitor=True, 
            save_monitor_stats_to_disk=False,
            save_config=True):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if config is None else get_env_stats(config.env)
        
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
    
    def build_strategy_from_scratch(self, 
            config=None, 
            build_monitor=True, 
            save_monitor_stats_to_disk=False,
            save_config=True):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if config is None else get_env_stats(config.env)
        
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

    def build_acting_agent_from_scratch(self, 
            config=None, 
            build_monitor=True, 
            to_build_for_eval=False):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if config is None else get_env_stats(config.env)
        
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
            monitor=None if build_monitor is None else elements.monitor, 
            config=config,
            constructors=constructors)

        return elements
    
    def build_training_agent_from_scratch(self, 
            config=None, 
            save_monitor_stats_to_disk=True,
            save_config=True):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if config is None else get_env_stats(config.env)
        
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

    def build_agent_from_scratch(self, 
            config=None, 
            save_monitor_stats_to_disk=True,
            save_config=True):
        constructors = self.get_constructors(config)
        env_stats = self.env_stats if config is None else get_env_stats(config.env)
        
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

    def get_constructors(self, config):
        if config is not None and config.algorithm != self.config.algorithm:
            constructors = self.retrieve_constructor(config.algorithm)
        else:
            constructors = self.constructors
        return constructors

    """ Configuration Operations """
    def get_config(self):
        return self.config

    def save_config(self):
        save_config(self.config)

    """ Save & Restore """
    def restore(self):
        if os.path.exists(self._builder_path) and self._incremental_version:
            with open(self._builder_path, 'rb') as f:
                self._version, self._max_version = cloudpickle.load(f)

    def save(self):
        if self._incremental_version:
            with open(self._builder_path, 'wb') as f:
                cloudpickle.dump((self._version, self._max_version), f)

    """ Implementations"""
    def _import_element(self, name, algo=None, *, config=None, place=0):
        try:
            module = pkg.import_module(
                f'elements.{name}', algo=algo)
        except:
            module = pkg.import_module(
                f'elements.{name}', pkg='core')
        return module

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
