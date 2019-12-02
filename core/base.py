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
        restore(self.ckpt_manager, self.ckpt, self.ckpt_path)

    def save(self, steps=None, message='', print_terminal_info=True):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            global_steps: A tensor that records step
            steps: An int that assigns to global_steps. 
                If it's None, we leave global_steps unchanged
            message: optional message for print
        """
        save(self.ckpt_manager, self.global_steps, steps, message, 
            print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self.logger, config)

    def log(self, step, timing='Train', print_terminal_info=True):
        log(self.logger, self.writer, self.model_name, step, 
            timing=timing, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self.logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def log_summary(self, stats, step=None):
        log_summary(self.writer, stats, step=step)

    def store(self, **kwargs):
        store(self.logger, **kwargs)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        get_stats(self.logger, mean, std, min, max)
