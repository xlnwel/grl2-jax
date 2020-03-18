import collections
import numpy as np
import tensorflow as tf
import ray


DataFormat = collections.namedtuple('DataFormat', ('shape', 'dtype'))

class Dataset:
    def __init__(self, buffer, data_format, process_fn=lambda data: data, batch_size=False):
        """ Create a tf.data.Dataset for data retrieval
        
        Args:
            buffer: buffer, a callable object that stores data
            data_format: dict, whose keys are keys of returned data
            values are tuple (type, shape) that passed to 
            tf.data.Dataset.from_generator
        """
        self._buffer = buffer
        self._data_format = data_format
        assert isinstance(data_format, dict)
        self._iterator = self._prepare_dataset(
            buffer, data_format, process_fn, batch_size)

    def buffer_type(self):
        return self._buffer.buffer_type()

    def good_to_learn(self):
        return self._buffer.good_to_learn()
        
    def sample(self):
        return next(self.iterator)

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities(priorities, indices)

    def _prepare_dataset(self, buffer, data_format, process_fn, batch_size):
        with tf.name_scope('data'):
            types = {k: v.dtype for k, v in data_format.items()}
            shapes = {k: v.shape for k, v in data_format.items()}

            if not self.buffer_type().endswith('uniform'):
                types['IS_ratio'] = tf.float32
                types['saved_idxes'] = tf.int32
                shapes['IS_ratio'] = (None)
                shapes['saved_idxes'] = (None)

            ds = tf.data.Dataset.from_generator(self._sample, types, shapes)
            if batch_size:
                ds = ds.batch(batch_size, drop_remainder=True)
            ds = ds.map(map_func=process_fn, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            iterator = iter(ds)
        return iterator

    def _sample(self):
        while True:
            yield self._buffer.sample()

class RayDataset(Dataset):
    def buffer_type(self):
        return ray.get(self._buffer.buffer_type.remote())

    def _sample(self):
        while True:
            yield ray.get(self._buffer.sample.remote())

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities.remote(priorities, indices)
