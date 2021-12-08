import time
import threading

from core.elements.builder import ElementsBuilder
from core.remote.base import RayBase
from utility.utils import dict2AttrDict


class RemoteAgent(RayBase):
    def __init__(self, config, env_stats):
        super().__init__()
        self.config = dict2AttrDict(config)

        self.builder = ElementsBuilder(
            config, 
            env_stats, 
            incremental_version=True,
            start_version=0)
        elements = self.builder.build_agent_from_scratch()
        self.agent = elements.agent
        self.buffer = elements.buffer
        self.train_step = None

    def merge(self, train_step, data, n):
        # print('merge', train_step, self.train_step, self.buffer.ready(), n, self.buffer.size(), self.buffer.max_size())
        if train_step != self.train_step:
            return False
        if self.buffer.ready():
            return True
        self.buffer.merge(data, n)
        return self.buffer.ready()

    def get_version(self):
        return self.builder.get_version()

    def increase_version(self):
        self.builder.increase_version()
        model_path = self.builder.get_model_path()
        self.agent.reset_model_path(*model_path)
        return model_path

    def is_buffer_ready(self):
        print(self.buffer.size(), self.buffer.max_size())
        return self.buffer.ready()

    """ Get & Set """
    def get_model_path(self):
        return self.builder.get_model_path()

    def get_weights(self, opt_weights):
        return self.train_step, self.agent.get_weights(opt_weights=opt_weights)
    
    def set_weights(self, weights):
        self.agent.set_weights(weights)

    def get_train_step(self):
        return self.agent.get_train_step()

    def wait_for_train_step_update(self):
        # assert self.buffer.ready() or self.train_step is not None, (self.buffer.size(), self.buffer.max_size())
        while self.train_step is None:
            time.sleep(.01)
        train_step = self.train_step
        self.train_step = None
        return train_step
    
    def get_env_step(self):
        return self.agent.get_env_step()

    """ Bookkeeping """
    def store(self, stats):
        self.agent.store(**stats)

    def record(self, step):
        self.agent.set_env_step(step)
        self.agent.record(step=step)
        self.agent.save()

    def start_training(self):
        self._training_thread = threading.Thread(target=self._training, daemon=True)
        self._training_thread.start()

    def _training(self):
        while True:
            step = self.agent.get_train_step()
            print('before train step', step)
            self.agent.train_record()
            self.train_step = self.agent.get_train_step()
            assert self.train_step - step == 8, (self.train_step, step)
