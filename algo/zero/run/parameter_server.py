import cloudpickle

from distributed.remote.base import RayBase
from utility.utils import config_attr


class ParameterServer(RayBase):
    def __init__(self, config):
        self.config = config_attr(self, config)
        self.path = f'{config.root_dir}/{config.model_name}/parameter_server.pkl'
        # {log dir: weights}
        self._agents = {}
        self.restore()
    
    def add_strategy(self, model_dir):
        self._agents.append(model_dir)
        self.save()

    def save(self):
         with open(self.path, 'wb') as f:
            cloudpickle.dump(self._agents, f)

    def restore(self):
        with open(self.path, 'rb') as f:
            self._agents = cloudpickle.load(f)
