from core.dataset import create_dataset
from core.monitor import create_monitor
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

        algo = config.algorithm
        self.create_model = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.create_loss = pkg.import_module(
                name='elements.loss', algo=algo, place=-1).create_loss
        self.create_trainer = pkg.import_module(
            name='elements.trainer', algo=algo, place=-1).create_trainer
        self.create_actor = pkg.import_module(
            name='elements.actor', algo=algo, place=-1).create_actor
        self.create_buffer = pkg.import_module(
            'elements.buffer', algo=self.config.algorithm).create_buffer
        self.create_strategy = pkg.import_module(
            'elements.strategy', algo=self.config.algorithm).create_strategy
        self.create_agent = pkg.import_module(
            'elements.agent', algo=self.config.algorithm).create_agent

    def build_model(self, to_build=False, to_build_for_eval=False):
        model = self.create_model(
            self.config.model, self.env_stats, 
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
    
    def build_buffer(self, model, buffer_config=None):
        if buffer_config is not None:
            self.config.buffer = buffer_config
        else:
            self.config.buffer['n_envs'] = self.env_stats.n_envs
            self.config.buffer['state_keys'] = model.state_keys
            self.config.buffer['use_dataset'] = self.config.buffer.get('use_dataset', False)
        buffer = self.create_buffer(self.config.buffer)
        
        return buffer

    def build_dataset(self, buffer, model):        
        if self.config.buffer['use_dataset']:
            am = pkg.import_module('elements.utils', algo=self.config.algorithm)
            data_format = am.get_data_format(
                self.config.trainer, self.env_stats, model)
            dataset = create_dataset(buffer, self.env_stats, 
                data_format=data_format, one_hot_action=False)
        else:
            dataset = buffer

        return dataset
    
    def build_strategy(self, actor=None, trainer=None, dataset=None):
        strategy = self.create_strategy(
            self.name, self.config.strategy, actor=actor, 
            trainer=trainer, dataset=dataset)
        
        return strategy

    def build_monitor(self):
        return create_monitor(self.config.root_dir, self.config.model_name, self.name)
    
    def build_agent(self, strategy, monitor=None, to_save_code=True):
        agent = self.create_agent(
            config=self.config.agent, 
            strategy=strategy, 
            monitor=monitor, 
            name=self.name, 
            to_save_code=to_save_code
        )

        return agent

    def get_config(self):
        return self.config
