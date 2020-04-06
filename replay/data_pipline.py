import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision.experimental import global_policy
import ray


DataFormat = collections.namedtuple('DataFormat', ('shape', 'dtype'))


def process_with_env(data, env, obs_range=[0, 1]):
    dtype = global_policy().compute_dtype
    with tf.device('cpu:0'):
        if env.obs_dtype == np.uint8:
            if obs_range == [0, 1]:
                data['obs'] = tf.cast(data['obs'], dtype) / 255.
            elif obs_range == [-.5, .5]:
                data['obs'] = tf.cast(data['obs'], dtype) / 255. - .5
            else:
                raise NotImplementedError(f'Unknown range: {obs_range}')
        if env.is_action_discrete:
            data['action'] = tf.one_hot(data['action'], env.action_dim, dtype=dtype)
    return data

class Dataset:
    def __init__(self, 
                 buffer, 
                 data_format, 
                 process_fn=None, 
                 batch_size=False, 
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
        self._iterator = self._prepare_dataset(process_fn, batch_size, **kwargs)

    def buffer_type(self):
        return self._buffer.buffer_type()

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

            if self.buffer_type().endswith('proportional'):
                types['IS_ratio'] = tf.float32
                types['saved_idxes'] = tf.int32
                shapes['IS_ratio'] = (None)
                shapes['saved_idxes'] = (None)

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
    def buffer_type(self):
        return ray.get(self._buffer.buffer_type.remote())

    def good_to_learn(self):
        return ray.get(self._buffer.good_to_learn.remote())

    def _sample(self):
        while True:
            yield ray.get(self._buffer.sample.remote())

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities.remote(priorities, indices)
