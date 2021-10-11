import os, atexit
import logging
from collections import defaultdict
import numpy as np
import tensorflow as tf

from core.log import do_logging
from utility.display import pwc
from utility.graph import image_summary, video_summary
from utility.utils import isscalar


logger = logging.getLogger(__name__)


""" Recorder """
class Recorder:
    def __init__(self, recorder_dir=None, record_file='record.txt'):
        """
        Initialize a Recorder.

        Args:
            log_dir (string): A directory for saving results to. If 
                `None/False`, Recorder only serves as a storage but 
                doesn't write anything to the disk.

            record_file (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to "log.txt". 
        """
        record_file = record_file if record_file.endswith('record.txt') \
            else record_file + '/record.txt'
        self._recorder_dir = recorder_dir
        if self._recorder_dir:
            path = os.path.join(self._recorder_dir, record_file)
            # if os.path.exists(path) and os.stat(path).st_size != 0:
            #     i = 1
            #     name, suffix = path.rsplit('.', 1)
            #     while os.path.exists(name + f'{i}.' + suffix):
            #         i += 1
            #     pwc(f'Warning: Log file "{path}" already exists!', 
            #         f'Data will be logged to "{name + f"{i}." + suffix}" instead.',
            #         color='magenta')
            #     path = name + f"{i}." + suffix
            if not os.path.isdir(self._recorder_dir):
                os.makedirs(self._recorder_dir)
            self._out_file = open(path, 'a')
            atexit.register(self._out_file.close)
            pwc(f'Record data to "{self._out_file.name}"', color='green')
        else:
            self._out_file = None
            pwc(f'Record directory is not specified, '
                'no data will be recorded to the disk',
                color='magenta')

        self._first_row=True
        self._headers = []
        self._current_row = {}
        self._store_dict = defaultdict(list)

    def __contains__(self, item):
        return item in self._store_dict and self._store_dict[item] != []
    
    def contains_stats(self, item):
        return item in self._store_dict and self._store_dict[item] != []
        
    def store(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            if v is None:
                return
            elif isinstance(v, (list, tuple)):
                self._store_dict[k] += list(v)
            else:
                self._store_dict[k].append(v)

    """ All get functions below will remove the corresponding items from the store """
    def get_raw_item(self, key):
        if key in self._store_dict:
            v = self._store_dict[key]
            del self._store_dict[key]
            return {key: v}
        return None
        
    def get_item(self, key, mean=True, std=False, min=False, max=False):
        stats = {}
        if key not in self._store_dict:
            return stats
        v = self._store_dict[key]
        if isscalar(v):
            stats[key] = v
            return
        if mean:
            stats[f'{key}'] = np.mean(v).astype(np.float32)
        if std:
            stats[f'{key}_std'] = np.std(v).astype(np.float32)
        if min:
            stats[f'{key}_min'] = np.min(v).astype(np.float32)
        if max:
            stats[f'{key}_max'] = np.max(v).astype(np.float32)
        del self._store_dict[key]
        return stats

    def get_raw_stats(self):
        stats = self._store_dict.copy()
        self._store_dict.clear()
        return stats

    def get_stats(self, mean=True, std=False, min=False, max=False):
        stats = {}
        for k in sorted(self._store_dict):
            v = self._store_dict[k]
            k_std, k_min, k_max = std, min, max
            if k.startswith('train/') or k.startswith('stats/'):
                k_std = k_min = k_max = True
            if isscalar(v):
                stats[k] = v
                continue
            if mean:
                stats[f'{k}'] = np.mean(v).astype(np.float32)
            if k_std:
                stats[f'{k}_std'] = np.std(v).astype(np.float32)
            if k_min:
                stats[f'{k}_min'] = np.min(v).astype(np.float32)
            if k_max:
                stats[f'{k}_max'] = np.max(v).astype(np.float32)
        self._store_dict.clear()
        return stats

    def get_count(self, name):
        return len(self._store_dict[name])

    def record_stats(self, stats, print_terminal_info=True):
        if not self._first_row and not set(stats).issubset(set(self._headers)):
            if self._first_row:
                do_logging(f'All previous records are erased because stats does not match the first row\n'
                    f'stats = {set(stats)}\nfirst row = {set(self._headers)}', 
                    logger=logger, level='WARNING')
            self._out_file.seek(0)
            self._out_file.truncate()
            self._headers.clear()
            self._first_row = True
        [self.record_tabular(k, v) for k, v in stats.items()]
        self.dump_tabular(print_terminal_info=print_terminal_info)

    def _record_tabular(self, key, val):
        """
        Record a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``record_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self._first_row:
            self._headers.append(key)
        else:
            assert key in self._headers, \
                f"Trying to introduce a new key {key} " \
                "that you didn't include in the first iteration"
        assert key not in self._current_row, \
            f"You already set {key} this iteration. " \
            "Maybe you forgot to call dump_tabular()"
        self._current_row[key] = val
    
    def record_tabular(self, key, val=None, mean=True, std=False, min=False, max=False):
        """
        Record a value or possibly the mean/std/min/max values of a diagnostic.
        """
        if val is not None:
            self._record_tabular(key, val)
        else:
            v = np.asarray(self._store_dict[key])
            if mean:
                self._record_tabular(f'{key}_mean', np.mean(v))
            if std:
                self._record_tabular(f'{key}_std', np.std(v))
            if min:
                self._record_tabular(f'{key}_min', np.min(v))
            if max:
                self._record_tabular(f'{key}_max', np.max(v))
        self._store_dict[key] = []

    def dump_tabular(self, print_terminal_info=True):
        """
        Write all of the diagnostics from the current iteration.
        """
        vals = []
        key_lens = [len(key) for key in self._headers]
        max_key_len = max(15,max(key_lens))
        n_slashes = 22 + max_key_len
        if print_terminal_info:
            print("-"*n_slashes)
        for key in self._headers:
            val = self._current_row.get(key, "")
            # print(key, np.array(val).dtype)
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else val
            if print_terminal_info:
                print(f'| {key:>{max_key_len}s} | {valstr:>15s} |')
            vals.append(val)
        if print_terminal_info:
            print("-"*n_slashes)
        if self._out_file is not None:
            if self._first_row:
                self._out_file.write("\t".join(self._headers)+"\n")
            self._out_file.write("\t".join(map(str,vals))+"\n")
            self._out_file.flush()
        self._current_row.clear()
        self._store_dict.clear()
        self._first_row=False


""" Tensorboard Writer """
class TensorboardWriter:
    def __init__(self, root_dir, model_name, name):
        self._writer = create_tb_writer(root_dir, model_name)
        self.name = name
        tf.summary.experimental.set_step(0)
    
    def set_summary_step(self, step):
        """ Sets tensorboard step """
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        """ Adds scalar summary to tensorboard """
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        """ Adds histogram summary to tensorboard """
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """ Adds graph summary to tensorboard
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def video_summary(self, video, step=None):
        video_summary(f'{self.name}/sim', video, step=step)

    def flush(self):
        self._writer.flush()

""" Recorder Ops """
def record_stats(recorder, stats, print_terminal_info=True):
    [recorder.record_tabular(k, v) for k, v in stats.items()]
    recorder.dump_tabular(print_terminal_info=print_terminal_info)

def store(recorder, **kwargs):
    recorder.store(**kwargs)

def get_raw_item(recorder, key):
    return recorder.get_raw_item(key)

def get_item(recorder, key, mean=True, std=False, min=False, max=False):
    return recorder.get_item(key, mean=mean, std=std, min=min, max=max)

def get_raw_stats(recorder):
    return recorder.get_raw_stats()

def get_stats(recorder, mean=True, std=False, min=False, max=False):
    return recorder.get_stats(mean=mean, std=std, min=min, max=max)

def contains_stats(recorder, key):
    return key in recorder

def create_recorder(root_dir, model_name):
    recorder_dir = root_dir and f'{root_dir}/{model_name}'
    # recorder save stats in f'{root_dir}/{model_name}/logs/record.txt'
    recorder = Recorder(recorder_dir)
    return recorder


""" Tensorboard Ops """
def set_summary_step(step):
    tf.summary.experimental.set_step(step)

def scalar_summary(writer, stats, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    prefix = prefix or 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if '/' not in k:
                k = f'{prefix}/{k}'
            # print(k, np.array(v).dtype)
            tf.summary.scalar(k, tf.reduce_mean(v), step=step)

def histogram_summary(writer, stats, prefix=None, step=None):
    if step is not None:
        tf.summary.experimental.set_step(step)
    prefix = prefix or 'stats'
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, (str, int, float)):
                continue
            tf.summary.histogram(f'{prefix}/{k}', v, step=step)

def graph_summary(writer, sum_type, args, step=None):
    """ This function should only be called inside a tf.function """
    fn = {'image': image_summary, 'video': video_summary}[sum_type]
    if step is None:
        step = tf.summary.experimental.get_step()
    def inner(*args):
        tf.summary.experimental.set_step(step)
        with writer.as_default():
            fn(*args)
    return tf.numpy_function(inner, args, [])

def create_tb_writer(root_dir, model_name):
    # writer for tensorboard summary
    # stats are saved in directory f'{root_dir}/{model_name}'
    writer = tf.summary.create_file_writer(
        f'{root_dir}/{model_name}', max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    return writer

def create_tensorboard_writer(root_dir, model_name, name):
    return TensorboardWriter(root_dir, model_name, name)
