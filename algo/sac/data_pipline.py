import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self, buffer, state_shape, action_dim):
        """ Create a tf.data.Dataset for data retrieval
        
        Args:
            buffer: buffer, a callable object that stores data
        """
        self.buffer = buffer
        self.iterator = self._prepare_dataset(buffer, state_shape, action_dim)

    @property
    def type(self):
        return self.buffer.type

    def get_data(self):
        return next(self.iterator)

    def update_priorities(self, priorities, indices):
        self.buffer.update_priorities(np.squeeze(priorities), indices)

    def _prepare_dataset(self, buffer, state_shape, action_dim):
        with tf.name_scope('data'):
            sample_types = (
                tf.float32, 
                tf.float32, 
                tf.float32, 
                tf.float32, 
                tf.float32, 
                tf.float32
            )
            sample_shapes = (
                (None, *state_shape),
                (None, action_dim),
                (None, 1),
                (None, *state_shape),
                (None, 1),
                (None, 1)
            )
            if buffer.type != 'uniform':
                sample_types = (tf.float32, tf.int32, sample_types)
                sample_shapes =((None), (None), sample_shapes)

            ds = tf.data.Dataset.from_generator(
                buffer, output_types=sample_types, output_shapes=sample_shapes)
            ds = ds.map(map_func=self._transform_data_per if buffer.type != 'uniform' 
                else self._transform_data_uniform, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            iterator = iter(ds)
        return iterator


    def _transform_data_per(self, IS_ratio, saved_indices, transition):
        data = {}

        state, action, reward, next_state, done, steps = transition

        data['IS_ratio'] = IS_ratio[:, None]        # Importance sampling ratio for PER
        # saved indexes used to index the experience in the buffer when updating priorities
        data['saved_indices'] = saved_indices

        data['state'] = state
        data['action'] = action
        data['reward'] = reward
        data['next_state'] = next_state
        data['done'] = done
        data['steps'] = steps

        return data

    def _transform_data_uniform(self, state, action, reward, next_state, done, steps):
        data = dict(
            IS_ratio=1  # fake ratio to avoid complicate the code
        )

        data['state'] = state
        data['action'] = action
        data['reward'] = reward
        data['next_state'] = next_state
        data['done'] = done
        data['steps'] = steps

        return data
