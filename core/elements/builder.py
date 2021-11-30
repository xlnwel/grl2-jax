from core.dataset import create_dataset
from core.monitor import create_monitor
from utility.typing import AttrDict
from utility.utils import dict2AttrDict
from utility import pkg


class ElementsBuilder:
    def __init__(self, 
                 config, 
                 env_stats, 
                 name):
        self.config = dict2AttrDict(config)
        self.env_stats = dict2AttrDict(env_stats)
        self.name = name

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

    """ Build Elements """
    def build_model(self, to_build=False, to_build_for_eval=False):
        model = self.create_model(
            self.config.model, self.env_stats, name=self.name,
            to_build=to_build, to_build_for_eval=to_build_for_eval)
        
        return model

    def build_actor(self, model):
        actor = self.create_actor(self.config.actor, model, name=self.name)
        
        return actor
    
    def build_trainer(self, model):
        loss = self.create_loss(self.config.loss, model, name=self.name)
        trainer = self.create_trainer(
                self.config.trainer, loss, self.env_stats, name=self.name)
        
        return trainer
    
    def build_buffer(self, model, buffer_config=None, central_buffer=False):
        if buffer_config is not None:
            self.config.buffer = buffer_config
        else:
            self.config.buffer['n_envs'] = self.env_stats.n_envs
            self.config.buffer['state_keys'] = model.state_keys
            self.config.buffer['use_dataset'] = self.config.buffer.get('use_dataset', False)
        buffer = self.create_buffer(self.config.buffer, central_buffer=central_buffer)
        
        return buffer

    def build_dataset(self, buffer, model, central_buffer=False):        
        if self.config.buffer['use_dataset']:
            am = pkg.import_module('elements.utils', algo=self.config.algorithm)
            data_format = am.get_data_format(
                self.config.trainer, self.env_stats, model)
            dataset = create_dataset(buffer, self.env_stats, 
                data_format=data_format, central_buffer=central_buffer, 
                one_hot_action=False)
        else:
            dataset = buffer

        return dataset
    
    def build_strategy(self, actor=None, trainer=None, dataset=None):
        strategy = self.create_strategy(
            self.name, self.config.strategy, actor=actor, 
            trainer=trainer, dataset=dataset)
        
        return strategy

    def build_monitor(self, save_to_disk=True):
        if save_to_disk:
            return create_monitor(self.config.root_dir, self.config.model_name, self.name)
        else:
            return create_monitor(None, None, self.name, use_tensorboard=False)
    
    def build_agent(self, strategy, monitor=None, to_save_code=True):
        agent = self.create_agent(
            config=self.config.agent, 
            strategy=strategy, 
            monitor=monitor, 
            name=self.name, 
            to_save_code=to_save_code
        )

        return agent

    """ Get configurations """
    def get_config(self):
        return self.config

    """ Build an Agent from Scratch """
    def build_actor_agent_from_scratch(self):
        elements = AttrDict()
        elements.model = self.build_model(to_build=True)
        elements.actor = self.build_actor(elements.model)
        elements.strategy = self.build_strategy(actor=elements.actor)
        elements.monitor = self.build_monitor()
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor, 
            to_save_code=False)

        return elements
    
    def build_agent_from_scratch(self):
        elements = AttrDict()
        elements.model = self.build_model()
        elements.actor = self.build_actor(elements.model)
        elements.trainer = self.build_trainer(elements.model)
        elements.buffer = self.build_buffer(elements.model)
        elements.dataset = self.build_dataset(elements.buffer, elements.model)
        elements.strategy = self.build_strategy(
            actor=elements.actor, 
            trainer=elements.trainer, 
            dataset=elements.dataset)
        elements.monitor = self.build_monitor()
        elements.agent = self.build_agent(
            strategy=elements.strategy, 
            monitor=elements.monitor)

        return elements
