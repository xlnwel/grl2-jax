import numpy as np
import tensorflow as tf
import ray


class Dataset:
    def __init__(self, buffer, data_format):
        """ Create a tf.data.Dataset for data retrieval
        
        Args:
            buffer: buffer, a callable object that stores data
            data_format: dict, whose keys are keys of returned data
            values are tuple (type, shape) that passed to 
            tf.data.Dataset.from_generator
        """
        self.buffer = buffer
        assert isinstance(data_format, dict)
        self.iterator = self._prepare_dataset(buffer, data_format)

    def buffer_type(self):
        return self.buffer.buffer_type()

    def good_to_learn(self):
        return self.buffer.good_to_learn()
        
    def sample(self):
        return next(self.iterator)

    def update_priorities(self, priorities, indices):
        self.buffer.update_priorities(priorities, indices)

    def _prepare_dataset(self, buffer, data_format):
        def process_transition(data):
            if data['state'].dtype == tf.uint8:
                tf.debugging.assert_shapes([(data['state'], (None, 84, 84, 4)), (data['next_state'], (None, 84, 84, 4))])
                tf.debugging.assert_type(data['state'], tf.uint8)
                tf.debugging.assert_type(data['next_state'], tf.uint8)
                data['state'] = tf.cast(data['state'], tf.float32) / 255.
                data['next_state'] = tf.cast(data['next_state'], tf.float32) / 255.

            for k in ['reward', 'done', 'steps']:
                data[k] = tf.expand_dims(data[k], -1)

                tf.debugging.assert_shapes([(data[k], (None, 1))])
                
            return data

        def transform_data_per(data):
            return process_transition(data)

        def transform_data_uniform(data):
            data['IS_ratio'] = 1. # fake ratio to avoid complicate the code

            return process_transition(data)

        with tf.name_scope('data'):
            sample_types = dict((k, v[0]) for k, v in data_format.items())
            sample_shapes = dict((k, v[1]) for k, v in data_format.items())

            if not self.buffer_type().endswith('uniform'):
                sample_types['IS_ratio'] = tf.float32
                sample_types['saved_indices'] = tf.int32
                sample_shapes['IS_ratio'] = (None, 1)
                sample_shapes['saved_indices'] = (None)

            ds = tf.data.Dataset.from_generator(
                self._sample, output_types=sample_types, output_shapes=sample_shapes)
            ds = ds.map(map_func=transform_data_uniform if self.buffer_type().endswith('uniform') 
                        else transform_data_per, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            iterator = iter(ds)
        return iterator

    def _sample(self):
        while True:
            yield self.buffer.sample()

class RayDataset(Dataset):
    def buffer_type(self):
        return ray.get(self.buffer.buffer_type.remote())

    def _sample(self):
        while True:
            yield ray.get(self.buffer.sample.remote())

    def update_priorities(self, priorities, indices):
        self.buffer.update_priorities.remote(priorities, indices)
