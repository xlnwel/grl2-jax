import os, time, atexit, shutil
from collections import defaultdict
import numpy as np
import tensorflow as tf

from utility.utils import isscalar, step_str
from utility.display import pwc, assert_colorize
from utility import yaml_op


""" Logging """
def save_config(logger, config):
    logger.save_config(config)

def log(logger, writer, model_name, step, timing='Train', print_terminal_info=True):
    stats = dict(
        model_name=f'{model_name}',
        timing=timing,
        steps=step
    )
    stats.update(logger.get_stats())
    log_summary(writer, stats, step)
    log_stats(logger, stats, print_terminal_info=print_terminal_info)

def log_stats(logger, stats, print_terminal_info=True):
    [logger.log_tabular(k, v) for k, v in stats.items()]
    logger.dump_tabular(print_terminal_info=print_terminal_info)

def set_summary_step(step):
    tf.summary.experimental.set_step(step)

def log_summary(writer, stats, step=None):
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if tf.rank(v).numpy() == 0:
                tf.summary.scalar(f'stats/{k}', v, step=step)
            else:
                v = tf.convert_to_tensor(v, dtype=tf.float32)
                tf.summary.scalar(f'stats/{k}_mean', tf.reduce_mean(v), step=step)
                tf.summary.scalar(f'stats/{k}_std', tf.math.reduce_std(v), step=step)

def store(logger, **kwargs):
    logger.store(**kwargs)

def get_stats(logger, mean=True, std=False, min=False, max=False):
    return logger.get_stats(mean=mean, std=std, min=min, max=max)

def get_value(logger, key, mean=True, std=False, min=False, max=False):
    return logger.get(key, mean=mean, std=std, min=min, max=max)

""" Functions for setup logging """                
def setup_logger(root_dir, model_name):
    # logger save stats in f'{root_dir}/{model_name}/logs/log.txt'
    logger = Logger(f'{root_dir}/{model_name}/logs')
    return logger

def setup_tensorboard(root_dir, model_name):
    # writer for tensorboard summary
    # stats are saved in directory f'{root_dir}/{model_name}'
    writer = tf.summary.create_file_writer(
        f'{root_dir}/{model_name}/logs', max_queue=100)
    writer.set_as_default()
    return writer

def save_code(root_dir, model_name):
    dest_dir = f'{root_dir}/{model_name}/src'
    shutil.copytree('.', dest_dir, ignore=shutil.ignore_patterns('logs', '.*'))

class Logger:
    def __init__(self, log_dir, log_file='log.txt'):
        """
        Initialize a Logger.

        Args:
            log_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            log_file (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        log_file = log_file if log_file.endswith('log.txt') else log_file + '/log.txt'
        self.log_dir = log_dir or f"/tmp/experiments/{time.time()}"
        path = os.path.join(self.log_dir, log_file)
        if os.path.exists(path):
            pwc(f'Warning: Log dir "{self.log_dir}" already exists!', 
                f'Overwrite or Append (o/a)?',
                color='magenta')
            ans = input()
            if ans.lower() == 'o':
                self.output_file = open(path, 'w')
                pwc(f'"{self.output_file.name}" will be OVERWRITTEN', color='magenta')
            else:
                self.output_file = open(path, 'a')
                pwc(f'New data will be appended to "{self.output_file.name}"', color='magenta')
        else:
            if not os.path.isdir(self.log_dir):
                os.makedirs(self.log_dir)
            self.output_file = open(path, 'w')
        atexit.register(self.output_file.close)
        pwc(f'Logging data to "{self.output_file.name}"', color='green')

        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.store_dict = defaultdict(list)

    def save_config(self, config, log_file='config.yaml'):
        yaml_op.save_config(config, filename=f'{self.log_dir}/{log_file}')

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            self.store_dict[k].append(v)

    def get(self, key, mean=True, std=False, min=False, max=False):
        stats = {}
        v = self.store_dict[key]
        if isscalar(v):
            stats[key] = v
            return
        if mean:
            stats[f'{key}'] = np.mean(v)
        if std:
            stats[f'{key}_std'] = np.std(v)
        if min:
            stats[f'{key}_min'] = np.min(v)
        if max:
            stats[f'{key}_max'] = np.max(v)
        del self.store_dict[key]
        return stats
        
    def get_stats(self, mean=True, std=False, min=False, max=False):
        stats = {} 
        for k, v in self.store_dict.items():
            if isscalar(v):
                stats[k] = v
                continue
            if mean:
                stats[f'{k}'] = np.mean(v)
            if std:
                stats[f'{k}_std'] = np.std(v)
            if min:
                stats[f'{k}_min'] = np.min(v)
            if max:
                stats[f'{k}_max'] = np.max(v)
        return stats

    def get_count(self, name):
        return len(self.store_dict[name])

    def _log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert_colorize(key in self.log_headers, f"Trying to introduce a new key {key} that you didn't include in the first iteration")
        assert_colorize(key not in self.log_current_row, f"You already set {key} this iteration. Maybe you forgot to call dump_tabular()")
        self.log_current_row[key] = val
    
    def log_tabular(self, key, val=None, mean=True, std=False, min=False, max=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        """
        if val is not None:
            self._log_tabular(key, val)
        else:
            v = np.asarray(self.store_dict[key])
            if mean:
                self._log_tabular(f'{key}_mean', np.mean(v))
            if std:
                self._log_tabular(f'{key}_std', np.std(v))
            if min:
                self._log_tabular(f'{key}_min', np.min(v))
            if max:
                self._log_tabular(f'{key}_max', np.max(v))
        self.store_dict[key] = []

    def dump_tabular(self, print_terminal_info=True):
        """
        Write all of the diagnostics from the current iteration.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        n_slashes = 22 + max_key_len
        if print_terminal_info:
            print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else val
            if print_terminal_info:
                print(f'| {key:>{max_key_len}s} | {valstr:>15s} |')
            vals.append(val)
        if print_terminal_info:
            print("-"*n_slashes)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.store_dict.clear()
        self.first_row=False
