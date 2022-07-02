import tensorflow as tf


class Dummy(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(name='dummy')

    def __call__(self, x, **kwargs):
        return x

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
