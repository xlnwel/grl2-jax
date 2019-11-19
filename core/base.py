from abc import ABC
import os
import numpy as np
import tensorflow as tf

from utility.logger import Logger
from utility.display import display_var_info, pwc, assert_colorize
from utility.yaml_op import load_config, save_config
from env.gym_env import create_gym_env, GymEnv


class BaseAgent(ABC):
    """ Restore & save """
    def restore(self):
        path = self.ckpt_manager.latest_checkpoint
        self.ckpt.restore(path)
        if path:
            pwc(f'Params for {self.name} are restored from "{path}".', color='cyan')
        else:
            pwc(f'No model for {self.name} is found at "{self.ckpt_path}"!', color='magenta')
            pwc(f'Continue or Exist (c/e):', color='magenta')
            ans = input()
            if ans.lower() == 'e':
                import sys
                sys.exit()
            else:
                pwc(f'Start training from scratch.', color='magenta')

    def save(self, steps, message=''):
        self.global_steps.assign(steps)
        path = self.ckpt_manager.save()
        pwc(f'Model saved at {path} {message}', color='cyan')

    """ Logging """
    def log(self, step, timing):
        stats = dict(
                model_name=f'{self.model_name}',
                timing=timing,
                steps=f'{step}'
        )
        stats.update(self.get_stats())
        self.log_summary(step, stats)
        self.log_stats(stats)

    def log_stats(self, stats):
        [self.logger.log_tabular(k, v) for k, v in stats.items()]
        self.logger.dump_tabular()

    def set_summary_step(self, step):
        tf.summary.experimental.set_step(step)

    def log_summary(self, step, stats):
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if tf.rank(v).numpy() == 0:
                tf.summary.scalar(f'stats/{k}', v)
            else:
                v = tf.convert_to_tensor(v, dtype=tf.float32)
                tf.summary.scalar(f'stats/{k}_mean', tf.reduce_mean(v))
                tf.summary.scalar(f'stats/{k}_std', tf.math.reduce_std(v))
        self.writer.flush()

    def store(self, **kwargs):
        self.logger.store(**kwargs)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return self.logger.get_stats(mean=mean, std=std, min=min, max=max)

    def get_value(self, key, mean=True, std=False, min=False, max=False):
        return self.logger.get(key)

    def log_tabular(self, key, value=None, mean=True, std=False, min_and_max=False):
        self.logger.log_tabular(key, value, mean, std, min_and_max)

    def dump_tabular(self, print_terminal_info=True):
        self.logger.dump_tabular(print_terminal_info=print_terminal_info)

    """ TF configurations """                
    def _setup_logger(self):
        # logger save stats in '{self.log_root_dir}/{self.model_name}-log.txt'
        self.logger = Logger(self.log_root_dir, self.model_name)

    def _setup_checkpoint(self, ckpt_models):
        # checkpoint & manager
        self.global_steps = tf.Variable(1)
        self.ckpt = tf.train.Checkpoint(step=self.global_steps, **ckpt_models)
        self.ckpt_path = f'{self.model_root_dir}/{self.model_name}'
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, 5)
    
    def _setup_tensorboard(self):
        # writer for tensorboard summary
        self.writer = tf.summary.create_file_writer(f'{self.log_root_dir}/{self.model_name}')
        self.writer.set_as_default()
        
    def _display_var_info(self):
        tvars = []
        for name, model in self.ckpt_models.items():
            if 'opt' in name:
                pass # ignore variables in the optimizer
            else:
                tvars += model.trainable_variables
            
        display_var_info(tvars)

    def _print_construction_complete(self):
        pwc(f'{self.name} has been constructed', color='cyan')


def agent_config(init_fn):
    """ Decorator for agent's initialization """
    from functools import wraps
    @wraps(init_fn)
    def wrapper(self, name, config, env=None, buffer=None, models=None, **kwargs):
        """
        Args:
            name: Agent's name
            config: configuration, should be read from config.yaml
            env: train environment
            buffer: buffer for transition storage
            models: a list of models that encapsulate network
            kwargs: optional arguments for each specific agent
        """
        # preprocessing
        self.name = name
        """ For the basic configuration, see config.yaml in algo/*/ """
        [setattr(self, k, v) for k, v in config.items()]

        self.env = env
        self.buffer = buffer

        self._setup_logger()
        self.logger.save_config(config)
        self._setup_tensorboard()

        # track models and optimizers for Checkpoint
        self.ckpt_models = {}
        if models:
            for m in models:
                setattr(self, m.name, m)
                self.ckpt_models[m.name] = m

        # initialization
        init_fn(self, name, config, env, buffer, models, **kwargs)
        
        # postprocessing
        self._setup_checkpoint(self.ckpt_models)
        self._display_var_info()
        self._print_construction_complete()
    
    return wrapper