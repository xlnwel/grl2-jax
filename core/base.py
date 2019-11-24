from abc import ABC

from core.log import *
from core.checkpoint import *


class BaseAgent(ABC):
    """ Restore & save """
    def restore(self):
        restore(self.ckpt_manager, self.ckpt, self.ckpt_path)

    def save(self, steps, message=''):
        save(self.ckpt_manager, self.global_steps, steps, message)

    """ Logging """
    def save_config(self, config):
        save_config(self.logger, config)

    def log(self, step, timing):
        log(self.logger, self.writer, self.model_name, step, timing)

    def log_stats(self, stats):
        log_stats(self.logger, stats)

    def set_summary_step(self, step):
        set_summary_step(step)

    def log_summary(self, stats, step=None):
        log_summary(self.writer, stats, step=step)

    def store(self, **kwargs):
        store(self.logger, **kwargs)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        get_stats(self.logger, mean, std, min, max)
