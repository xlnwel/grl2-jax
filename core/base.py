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

    def save(self, steps=None, message='', print_terminal_info=True):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            global_steps: A tensor that records step
            steps: An int that assigns to global_steps. 
                If it's None, we leave global_steps unchanged
            message: optional message for print
        """
        save(self._ckpt_manager, self.global_steps, steps, message, 
            print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, print_terminal_info=True):
        log(self._logger, self._writer, self._model_name, name=self.name, 
            step=step, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def scalar_summary(self, stats, step=None):
        scalar_summary(self._writer, stats, step=step, name=self.name)

    def graph_summary(self, fn=None, *args):
        graph_summary(self._writer, fn, *args)

    def store(self, **kwargs):
        store(self._logger, **kwargs)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return get_stats(self._logger, mean=mean, std=std, min=min, max=max)

    def get_value(self, key, mean=True, std=False, min=False, max=False):
        return get_value(self._logger, key, mean=mean, std=std, min=min, max=max)

    def print_construction_complete(self):
        pwc(f'{self._model_name.title()} is constructed...', color='cyan')
