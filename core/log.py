import os, atexit, shutil, datetime
from collections import defaultdict
import numpy as np
import tensorflow as tf

from utility.utils import isscalar, step_str
from utility.display import pwc, assert_colorize
from utility import yaml_op


""" Logging """
def log(logger, writer, model_name, name, step, print_terminal_info=True):
    stats = dict(
        model_name=f'{model_name}',
        steps=step
    )
    stats.update(logger.get_stats())
    scalar_summary(writer, stats, name=name, step=step)
    log_stats(logger, stats, print_terminal_info=print_terminal_info)

def log_stats(logger, stats, print_terminal_info=True):
    [logger.log_tabular(k, v) for k, v in stats.items()]
    logger.dump_tabular(print_terminal_info=print_terminal_info)

def set_summary_step(step):
    tf.summary.experimental.set_step(step)

def scalar_summary(writer, stats, name=None, step=None):
    set_summary_step(step)
    prefix = f'{name}/stats' if name else 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if tf.rank(v).numpy() == 0:
                tf.summary.scalar(f'{prefix}/{k}', v, step=step)
            else:
                v = tf.convert_to_tensor(v, dtype=tf.float32)
                tf.summary.scalar(f'{prefix}/{k}_mean', tf.reduce_mean(v), step=step)
                tf.summary.scalar(f'{prefix}/{k}_std', tf.math.reduce_std(v), step=step)

def graph_summary(writer, fn, *args):
    """ see utility.graph for available candidates of fn """
    step = tf.summary.experimental.get_step()
    def inner(*args):
        tf.summary.experimental.set_step(step)
        with writer.as_default():
            fn(*args)
    return tf.numpy_function(inner, args, [])

def store(logger, **kwargs):
    logger.store(**kwargs)

def get_stats(logger, mean=True, std=False, min=False, max=False):
    return logger.get_stats(mean=mean, std=std, min=min, max=max)

def get_value(logger, key, mean=True, std=False, min=False, max=False):
    return logger.get(key, mean=mean, std=std, min=min, max=max)

def save_code(root_dir, model_name):
    dest_dir = f'{root_dir}/{model_name}/src'
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    
    shutil.copytree('.', dest_dir, 
        ignore=shutil.ignore_patterns(
            '*logs*', '*data*', '.*', '*pycache*', '*.md', '*test*'))

def save_config(root_dir, model_name, config):
    yaml_op.save_config(config, filename=f'{root_dir}/{model_name}/config.yaml')

""" Functions for setup logging """                
def setup_logger(root_dir, model_name):
    log_dir = root_dir and f'{root_dir}/{model_name}/logs'
    # logger save stats in f'{root_dir}/{model_name}/logs/log.txt'
    logger = Logger(log_dir)
    return logger

def setup_tensorboard(root_dir, model_name):
    # writer for tensorboard summary
    # stats are saved in directory f'{root_dir}/{model_name}'
    writer = tf.summary.create_file_writer(
        f'{root_dir}/{model_name}/logs', max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    return writer

class Logger:
    def __init__(self, log_dir=None, log_file='log.txt'):
        """
        Initialize a Logger.

        Args:
            log_dir (string): A directory for saving results to. If 
                `None/False`, Logger only serves as a storage but doesn't
                write anything to the disk.

            log_file (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 
        """
        log_file = log_file if log_file.endswith('log.txt') \
            else log_file + '/log.txt'
        self._log_dir = log_dir
        if self._log_dir:
            path = os.path.join(self._log_dir, log_file)
            if os.path.exists(path):
                pwc(f'Warning: Log dir "{self._log_dir}" already exists!', 
                    f'Overwrite or Append (o/a)?',
                    color='magenta')
                ans = input()
                if ans.lower() == 'o':
                    self._out_file = open(path, 'w')
                    pwc(f'"{self._out_file.name}" will be OVERWRITTEN', 
                        color='magenta')
                else:
                    self._out_file = open(path, 'a')
                    pwc(f'New data will be appended to "{self._out_file.name}"', 
                        color='magenta')
            else:
                if not os.path.isdir(self._log_dir):
                    os.makedirs(self._log_dir)
                self._out_file = open(path, 'w')
            atexit.register(self._out_file.close)
            pwc(f'Logging data to "{self._out_file.name}"', color='green')
        else:
            self._out_file = None
            pwc(f'Log directory is not specified, '
                'no data will be written to the disk',
                color='magenta')

        self._first_row=True
        self._log_headers = []
        self._log_current_row = {}
        self._store_dict = defaultdict(list)

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            if isinstance(v, (list, tuple)):
                self._store_dict[k] += list(v)
            else:
                self._store_dict[k].append(v)

    def get(self, key, mean=True, std=False, min=False, max=False):
        stats = {}
        v = self._store_dict[key]
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
        del self._store_dict[key]
        return stats
        
    def get_stats(self, mean=True, std=False, min=False, max=False):
        stats = {} 
        for k, v in self._store_dict.items():
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
        return len(self._store_dict[name])

    def _log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self._first_row:
            self._log_headers.append(key)
        else:
            assert_colorize(key in self._log_headers, 
                f"Trying to introduce a new key {key} "
                "that you didn't include in the first iteration")
        assert_colorize(key not in self._log_current_row, 
            f"You already set {key} this iteration. "
            "Maybe you forgot to call dump_tabular()")
        self._log_current_row[key] = val
    
    def log_tabular(self, key, val=None, mean=True, std=False, min=False, max=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        """
        if val is not None:
            self._log_tabular(key, val)
        else:
            v = np.asarray(self._store_dict[key])
            if mean:
                self._log_tabular(f'{key}_mean', np.mean(v))
            if std:
                self._log_tabular(f'{key}_std', np.std(v))
            if min:
                self._log_tabular(f'{key}_min', np.min(v))
            if max:
                self._log_tabular(f'{key}_max', np.max(v))
        self._store_dict[key] = []

    def dump_tabular(self, print_terminal_info=True):
        """
        Write all of the diagnostics from the current iteration.
        """
        vals = []
        key_lens = [len(key) for key in self._log_headers]
        max_key_len = max(15,max(key_lens))
        n_slashes = 22 + max_key_len
        if print_terminal_info:
            print("-"*n_slashes)
        for key in self._log_headers:
            val = self._log_current_row.get(key, "")
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else val
            if print_terminal_info:
                print(f'| {key:>{max_key_len}s} | {valstr:>15s} |')
            vals.append(val)
        if print_terminal_info:
            print("-"*n_slashes)
        if self._out_file is not None:
            if self._first_row:
                self._out_file.write("\t".join(self._log_headers)+"\n")
            self._out_file.write("\t".join(map(str,vals))+"\n")
            self._out_file.flush()
        self._log_current_row.clear()
        self._store_dict.clear()
        self._first_row=False
