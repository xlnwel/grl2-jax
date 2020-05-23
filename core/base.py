from abc import ABC

from core.log import *
from core.checkpoint import *


class BaseAgent(ABC):
    """ Restore & save """
    def restore(self):
        """ Restore the latest parameter recorded by ckpt_manager

        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            ckpt: An instance of tf.train.Checkpoint
            ckpt_path: The directory in which to write checkpoints
            name: optional name for print
        """
        restore(self._ckpt_manager, self._ckpt, self._ckpt_path, self._model_name)
        self.env_steps = self._env_steps.numpy()
        self.train_steps = self._train_steps.numpy()

    def save(self, print_terminal_info=False):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
        """
        self._env_steps.assign(self.env_steps)
        self._train_steps.assign(self.train_steps)
        save(self._ckpt_manager, print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, prefix=None, print_terminal_info=True):
        prefix = prefix or self.name
        log(self._logger, self._writer, self._model_name, prefix=prefix, 
            step=step, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        prefix = prefix or self.name
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        prefix = prefix or self.name
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, fn=None, *args, step=None):
        graph_summary(self._writer, fn, *args, step=step)

    def store(self, **kwargs):
        store(self._logger, **kwargs)

    def get_raw_value(self, key):
        return get_raw_value(self._logger, key)

    def get_value(self, key, mean=True, std=False, min=False, max=False):
        return get_value(self._logger, key, mean=mean, std=std, min=min, max=max)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return get_stats(self._logger, mean=mean, std=std, min=min, max=max)

    def print_construction_complete(self):
        pwc(f'{self._model_name.title()} is constructed...', color='cyan')
