import tensorflow as tf


def select_optimizer(name):
    opts = dict(
        adam=tf.keras.optimizers.Adam,
        rmsprop=tf.keras.optimizers.RMSprop,
    )
    return opts[name.lower()]


class Optimizer(tf.Module):
    def __init__(self, name, models, lr, clip_norm=None, weight_decay=None, **kwargs):
        self._models = models if isinstance(models, (list, tuple)) else [models]
        self._clip_norm = clip_norm
        self._weight_decay = weight_decay
        self._opt = select_optimizer(name)(lr, **kwargs)
        # useful for mixed precision training on GPUs to
        # avoid numerical underflow caused by using float16 gradients
        self._opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self._opt, 'dynamic')
        self._variables = None

    @property
    def variables(self):
        return self._opt.variables()
    
    def __call__(self, tape, loss):
        if self._variables is None:
            variables = [m.trainable_variables for m in self._models]
            self._variables = tf.nest.flatten(variables)
        with tape:
            loss = self._opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self._variables)
        grads = self._opt.get_unscaled_gradients(grads)
        norm = tf.linalg.global_norm(grads)
        if self._clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm, norm)
        if self._weight_decay:
            context = tf.distribute.get_replica_context()
            context.merge_call(self._apply_weight_decay)
        self._opt.apply_gradients(zip(grads, self._variables))
        return norm