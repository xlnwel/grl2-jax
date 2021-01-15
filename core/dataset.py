import logging
import functools
import collections
import numpy as np
import tensorflow as tf
import ray


DataFormat = collections.namedtuple('DataFormat', ('shape', 'dtype'))
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, 
                 buffer, 
                 data_format, 
                 process_fn=None, 
                 batch_size=False, 
                 print_data_format=True,
                 **kwargs):
        """ Create a tf.data.Dataset for data retrieval
        
        Args:
            buffer: buffer, a callable object that stores data
            data_format: dict, whose keys are keys of returned data
            values are tuple (type, shape) that passed to 
            tf.data.Dataset.from_generator
        """
        self._buffer = buffer
        assert isinstance(data_format, dict)
        data_format = {k: DataFormat(*v) for k, v in data_format.items()}
        self.data_format = data_format
        if print_data_format:
            logger.info('Dataset info:')
            for k, v in data_format.items():
                logger.info(f'\t{k} {v}')
        self._iterator = self._prepare_dataset(process_fn, batch_size, **kwargs)

    def name(self):
        return self._buffer.name()

    def good_to_learn(self):
        return self._buffer.good_to_learn()
        
    def sample(self):
        return next(self._iterator)

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities(priorities, indices)

    def _prepare_dataset(self, process_fn, batch_size, **kwargs):
        with tf.name_scope('data'):
            types = {k: v.dtype for k, v in self.data_format.items()}
            shapes = {k: v.shape for k, v in self.data_format.items()}

            ds = tf.data.Dataset.from_generator(self._sample, types, shapes)
            if batch_size:
                ds = ds.batch(batch_size, drop_remainder=True)
            if process_fn:
                ds = ds.map(map_func=process_fn, 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
            prefetch = kwargs.get('prefetch', tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(prefetch)
            iterator = iter(ds)
        return iterator

    def _sample(self):
        while True:
            yield self._buffer.sample()

class RayDataset(Dataset):
    def name(self):
        return ray.get(self._buffer.name.remote())

    def good_to_learn(self):
        return ray.get(self._buffer.good_to_learn.remote())

    def _sample(self):
        while True:
            yield ray.get(self._buffer.sample.remote())

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities.remote(priorities, indices)

def process_with_env(data, env, obs_range=None, one_hot_action=True, dtype=tf.float32):
    with tf.device('cpu:0'):
        if env.obs_dtype == np.uint8 and obs_range is not None:
            if obs_range == [0, 1]:
                for k in data:
                    if 'obs' in k:
                        data[k] = tf.cast(data[k], dtype) / 255.
            elif obs_range == [-.5, .5]:
                for k in data:
                    if 'obs' in k:
                        data[k] = tf.cast(data[k], dtype) / 255. - .5
            else:
                raise ValueError(obs_range)
        if env.is_action_discrete and one_hot_action:
            for k in data:
                if 'action' in k:
                    data[k] = tf.one_hot(data[k], env.action_dim, dtype=dtype)
    return data


def create_dataset(replay, env, data_format=None, DatasetClass=Dataset, one_hot_action=True):
    if data_format is None:
        import time
        i = 0
        while not ray.get(replay.good_to_learn.remote()):
            time.sleep(1)
            i += 1
            if i % 60 == 0:
                size = ray.get(replay.size.remote())
                if size == 0:
                    import sys
                    print('Replay does not collect any data in 60s. Specify data_format for dataset construction explicitly')
                    sys.exit()
                print(f'Dataset Construction: replay size = {size}')
        data = ray.get(replay.sample.remote())
        data_format = {k: (v.shape, v.dtype) for k, v in data.items()}
        print('data format')
        for k, v in data_format.items():
            print('\t', k, v)
    process = functools.partial(process_with_env, 
        env=env, one_hot_action=one_hot_action)
    dataset = DatasetClass(replay, data_format, process)
    return dataset