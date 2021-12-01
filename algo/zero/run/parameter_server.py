import random
import cloudpickle

from distributed.remote.base import RayBase
from utility.utils import config_attr


class ParameterServer(RayBase):
    def __init__(self, config):
        self.config = config_attr(self, config)
        self.path = f'{config.root_dir}/{config.model_name}/parameter_server.pkl'
        self._agents = []
        self.restore()
    
    def add_strategy(self, path):
        if path not in self._agents:
            self._agents.append(path)
        self.save()

    def sample_strategy(self):
        return random.choice(self._agents)

    def save(self):
         with open(self.path, 'wb') as f:
            cloudpickle.dump(self._agents, f)

    def restore(self):
        with open(self.path, 'rb') as f:
            self._agents = cloudpickle.load(f)
