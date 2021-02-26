import re
import logging
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec


logger = logging.getLogger(__name__)

def select_optimizer(name):
    opts = dict(
        adam=tf.keras.optimizers.Adam,
        rmsprop=tf.keras.optimizers.RMSprop,
    )
    if isinstance(name, str):
        return opts[name.lower()]
    return name


class Optimizer(tf.Module):
    def __init__(self, name, models, lr, clip_norm=None, weight_decay=None, 
                wdpattern=r'.*', scales=None, return_grads=False, **kwargs):
        self._models = models if isinstance(models, (list, tuple)) else [models]
        self._clip_norm = clip_norm
        self._weight_decay = weight_decay
        self._wdpattern = wdpattern
        if scales is not None:
            assert isinstance(scales, (list, tuple)), scales
            assert len(scales) == len(self._models), (len(scales), len(self._models))
        self._scales = scales
        self._opt = select_optimizer(name)(lr, **kwargs)
        self._return_grads = return_grads
        # useful for mixed precision training on GPUs to
        # avoid numerical underflow caused by using float16 gradients
        prec_policy = prec.global_policy()
        self._mpt = prec_policy.compute_dtype != prec_policy.variable_dtype
        if self._mpt:
            logger.info('Mixed precision training will be performed')
            self._opt = prec.LossScaleOptimizer(self._opt)
        # we do not initialize variables here, as models may not be initialized at this point
        self._variables = None

    @property
    def variables(self):
        return self._opt.variables()
    
    def get_transformed_grads(self, var_list=[]):
        assert hasattr(self._opt, 'get_transformed_grads'), f'{self._opt} does not support "get_transformed_grads"'
        return self._opt.get_transformed_grads(var_list or self._variables)

    def __call__(self, tape, loss, output_gradients=None):
        if self._variables is None:
            variables = [m.trainable_variables for m in self._models]
            self._variables = tf.nest.flatten(variables)
            if self._scales is not None:
                scales = [[self._scales[i] for _ in m.trainable_variables] 
                    for i, m in enumerate(self._models)]
                self._scales = tf.nest.flatten(scales)
        if isinstance(loss, tf.Tensor):
            assert loss.shape == (), loss.shape
        if self._mpt:
            with tape:
                loss = self._opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self._variables, output_gradients=output_gradients)
        assert None not in grads, f'No grads for {self._variables[grads.index(None)].name}'
        if self._mpt:
            grads = self._opt.get_unscaled_gradients(grads)
        if self._scales is not None:
            assert len(grads) == len(self._scales), (len(grads), len(self._scales))
            grads = [g * s for g, s in zip(grads, self._scales)]
        norm = tf.linalg.global_norm(grads)
        if self._clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm, norm)
        if self._weight_decay:
            context = tf.distribute.get_replica_context()
            context.merge_call(self._apply_weight_decay)
        self.grads = grads
        self._opt.apply_gradients(zip(grads, self._variables))

        if self._return_grads:
            return norm, {v.name: g for v, g in zip(self._variables, grads)}
        else:
            return norm
    
    def _apply_weight_decay(self, strategy):
        for var in self._variables:
            if re.search(self._wdpattern, var.name):
                strategy.extended.update(var, lambda var: self._weight_decay * var)

if __name__ == '__main__':
    prec.set_global_policy('mixed_float16')
    policy = prec.global_policy()
    print(policy.compute_dtype)
    print(policy.variable_dtype)