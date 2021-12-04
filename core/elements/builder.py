from core.dataset import create_dataset
from core.elements.agent import RemoteAgent
from core.monitor import create_monitor
from core.utils import save_config
from run.utils import set_path
from utility.typing import AttrDict
from utility.utils import dict2AttrDict
from utility import pkg


class ElementsBuilder:
    def __init__(self, 
                 config, 
                 env_stats, 
                 name=None,
                 incremental_version=False,
                 start_version=0):
        # self.default_config = dict2AttrDict(config)
        self.config = dict2AttrDict(config)
        self.env_stats = dict2AttrDict(env_stats)
        self._name = name or self.config.name
        self._model_name = self.config.model_name
        self._incremental_version = incremental_version
        self._version = start_version
        if self._incremental_version:
            self.set_config_version(self._version)

        algo = self.config.algorithm
        self.create_model = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.create_loss = pkg.import_module(
                name='elements.loss', algo=algo, place=-1).create_loss
        self.create_trainer = pkg.import_module(
            name='elements.trainer', algo=algo, place=-1).create_trainer
        self.create_actor = pkg.import_module(
            name='elements.actor', algo=algo, place=-1).create_actor
        self.create_buffer = pkg.import_module(
            'elements.buffer', algo=algo).create_buffer
        self.create_strategy = pkg.import_module(
            'elements.strategy', algo=algo).create_strategy
        self.create_agent = pkg.import_module(
            'elements.agent', algo=algo).create_agent

    @property
    def name(self):
        return self._name

    def increase_version(self):
        self._version += 1
        self.set_config_version(self._version)
        self.save_config()

    def set_config_version(self, version):
        self.config = set_path(
            self.config, self.config.root_dir, f'{self._model_name}/v{version}')
        self.config['version'] = self._version = version

    """ Build Elements """
    def build_model(self, config=None, to_build=False, to_build_for_eval=False):
        config = dict2AttrDict(config or self.config)
        model = self.create_model(
            config.model, self.env_stats, name=self.name,
            to_build=to_build, to_build_for_eval=to_build_for_eval)
        
        return model

    def build_actor(self, model, config=None):
        config = dict2AttrDict(config or self.config)
        actor = self.create_actor(config.actor, model, name=self.name)
        
        return actor
    
    def build_trainer(self, model, config=None):
        config = dict2AttrDict(config or self.config)
        loss = self.create_loss(config.loss, model, name=self.name)
        trainer = self.create_trainer(
                config.trainer, loss, self.env_stats, name=self.name)
        
        return trainer
    
    def build_buffer(self, model, config=None, central_buffer=False):
        if config is None:
            self.config.buffer['n_envs'] = self.env_stats.n_envs
            self.config.buffer['state_keys'] = model.state_keys
            self.config.buffer['use_dataset'] = self.config.buffer.get('use_dataset', False)
        config = dict2AttrDict(config or self.config)
        buffer = self.create_buffer(config.buffer, central_buffer=central_buffer)
        
        return buffer

    def build_dataset(self, buffer, model, config=None, central_buffer=False):
        config = dict2AttrDict(config or self.config)
        if self.config.buffer['use_dataset']:
            am = pkg.import_module('elements.utils', algo=config.algorithm)
            data_format = am.get_data_format(
                self.config.trainer, self.env_stats, model)
            dataset = create_dataset(buffer, self.env_stats, 
                data_format=data_format, central_buffer=central_buffer, 
                one_hot_action=False)
        else:
            dataset = buffer

        return dataset
    
    def build_strategy(self, actor=None, trainer=None, dataset=None, config=None):
        config = dict2AttrDict(config or self.config)
        strategy = self.create_strategy(
            self.name, config.strategy, actor=actor, 
            trainer=trainer, dataset=dataset)
        
        return strategy

    def build_monitor(self, config=None, save_to_disk=True):
        config = dict2AttrDict(config or self.config)
        if save_to_disk:
            return create_monitor(config.root_dir, config.model_name, self.name)
        else:
            return create_monitor(None, None, self.name, use_tensorboard=False)
    
    def build_agent(self, strategy, monitor=None, config=None, to_save_code=True):
        config = dict2AttrDict(config or self.config)
        agent = self.create_agent(
            config=config.agent, 
            strategy=strategy, 
            monitor=monitor, 
            name=self.name, 
            to_save_code=to_save_code
        )

        return agent

    """ Build an Strategy/Agent from Scratch """
    def build_actor_strategy_from_scratch(self, config=None, build_monitor=True):
        elements = AttrDict()
        elements.model = self.build_model(config=config, to_build=True)
        elements.actor = self.build_actor(model=elements.model, config=config)
        elements.strategy = self.build_strategy(actor=elements.actor, config=config)
        if build_monitor:
            elements.monitor = self.build_monitor(config=config)

        return elements
    
    def build_strategy_from_scratch(self, config=None, build_monitor=True):
        elements = AttrDict()
        elements.model = self.build_model(config=config)
        elements.actor = self.build_actor(model=elements.model, config=config)
        elements.trainer = self.build_trainer(model=elements.model, config=config)
        elements.buffer = self.build_buffer(model=elements.model, config=config)
        elements.dataset = self.build_dataset(
            buffer=elements.buffer, 
            model=elements.model, 
            config=config)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            trainer=elements.trainer, 
            dataset=elements.dataset,
            config=config)
        if build_monitor:
            elements.monitor = self.build_monitor(config=config)

        return elements

    def build_actor_agent_from_scratch(self, config=None):
        elements = AttrDict()
        elements.model = self.build_model(config=config, to_build=True)
        elements.actor = self.build_actor(model=elements.model, config=config)
        elements.strategy = self.build_strategy(actor=elements.actor, config=config)
        elements.monitor = self.build_monitor(config=config)
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor, 
            config=config,
            to_save_code=False)

        return elements
    
    def build_agent_from_scratch(self, config=None):
        elements = AttrDict()
        elements.model = self.build_model(config=config)
        elements.actor = self.build_actor(model=elements.model, config=config)
        elements.trainer = self.build_trainer(model=elements.model, config=config)
        elements.buffer = self.build_buffer(model=elements.model, config=config)
        elements.dataset = self.build_dataset(
            buffer=elements.buffer, 
            model=elements.model, 
            config=config)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            trainer=elements.trainer, 
            dataset=elements.dataset,
            config=config)
        elements.monitor = self.build_monitor()
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor,
            config=config)
        
        self.save_config()
        if self._incremental_version:
            self.increase_version()

        return elements

    def build_remote_agent(self, config):
        agent = RemoteAgent.remote()
        agent.build(self, config)
        return agent

    """ Configuration Operations """
    def get_config(self):
        return self.config

    def save_config(self):
        save_config(self.config.root_dir, self.config.model_name, self.config)

    def get_model_path(self):
        return self.config.root_dir, self.config.model_name


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
