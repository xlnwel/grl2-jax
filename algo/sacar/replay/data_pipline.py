import numpy as np
import tensorflow as tf
import ray


class Dataset:
    def __init__(self, buffer, state_shape, state_dtype, action_shape, action_dtype):
        """ Create a tf.data.Dataset for data retrieval
        
        Args:
            buffer: buffer, a callable object that stores data
        """
        self.buffer = buffer
        self.iterator = self._prepare_dataset(
            buffer, state_shape, state_dtype, action_shape, action_dtype)

    def buffer_type(self):
        return self.buffer.buffer_type()

    def good_to_learn(self):
        return self.buffer.good_to_learn()
        
    def sample(self):
        return next(self.iterator)

    def update_priorities(self, priorities, indices):
        self.buffer.update_priorities(np.squeeze(priorities), indices)

    def _prepare_dataset(self, buffer, state_shape, state_dtype, action_shape, action_dtype):
        def transform_data_per(IS_ratio, saved_indices, transition):
            data = {}

            state, action, n_ar, reward, next_state, done, steps = transition

            if state.dtype == tf.uint8:
                state = tf.cast(state, tf.float32) / 255.
                next_state = tf.cast(next_state, tf.float32) / 255.

            data['IS_ratio'] = tf.expand_dims(IS_ratio, -1)        # Importance sampling ratio for PER
            # saved indexes used to index the experience in the buffer when updating priorities
            data['saved_indices'] = saved_indices

            data['state'] = state
            data['action'] = action
            data['n_ar'] = n_ar
            data['reward'] = tf.expand_dims(reward, -1)
            data['next_state'] = next_state
            data['done'] = tf.expand_dims(done, -1)
            data['steps'] = tf.expand_dims(steps, -1)

            return data

        def transform_data_uniform(state, action, n_ar, reward, next_state, done, steps):
            data = dict(
                IS_ratio=1.  # fake ratio to avoid complicate the code
            )
            
            if state.dtype == tf.uint8:
                state = tf.cast(state, tf.float32) / 255.
                next_state = tf.cast(next_state, tf.float32) / 255.
            
            data['state'] = state
            data['action'] = action
            data['n_ar'] = n_ar
            data['reward'] = tf.expand_dims(reward, -1)
            data['next_state'] = next_state
            data['done'] = tf.expand_dims(done, -1)
            data['steps'] = tf.expand_dims(steps, -1)

            return data

        with tf.name_scope('data'):
            sample_types = (
                state_dtype, 
                action_dtype, 
                tf.int32,
                tf.float32, 
                state_dtype, 
                tf.float32, 
                tf.float32
            )
            sample_shapes = (
                (None, *state_shape),
                (None, *action_shape),
                (None),
                (None), 
                (None, *state_shape),
                (None),
                (None)
            )

            if not self.buffer_type().endswith('uniform'):
                sample_types = (tf.float32, tf.int32, sample_types)
                sample_shapes =((None), (None), sample_shapes)

            ds = tf.data.Dataset.from_generator(
                self._sample, output_types=sample_types, output_shapes=sample_shapes)
            ds = ds.map(map_func=transform_data_uniform if self.buffer_type().endswith('uniform') 
                        else transform_data_per, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
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
        self.buffer.update_priorities.remote(np.squeeze(priorities), indices)
