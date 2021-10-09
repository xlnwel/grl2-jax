import re
import logging
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

from core.log import do_logging
from utility.schedule import TFPiecewiseSchedule


logger = logging.getLogger(__name__)

def select_optimizer(name):
    # add custom optimizers here
    opts = dict(
        adam=tf.keras.optimizers.Adam,
        rmsprop=tf.keras.optimizers.RMSprop,
    )
    if isinstance(name, str):
        return opts[name.lower()]
    return name


def create_optimizer(modules, config):
    if config.pop('schedule_lr', False):
        if not isinstance(config['lr'], (list, tuple)) \
                or not isinstance(config['lr'][0], (list, tuple)):
            raise ValueError(f"Require a list of tuples to schedule learning rate, but get lr={config['lr']}")
        config['lr'] = TFPiecewiseSchedule(config['lr'])
    do_logging(f'The optimizer for modules{tuple(m.name for m in modules)} is constructed with arguments:', logger=logger)
    do_logging(config, prefix='\t', logger=logger)
    opt = Optimizer(modules, **config)
    return opt


class Optimizer(tf.Module):
    def __init__(self, modules, *, opt_name='adam', lr, 
                clip_norm=None, weight_decay=None, l2_reg=None,
                wdpattern=r'.*', scales=None, return_grads=False, 
                **kwargs):
        self._modules = modules if isinstance(modules, (list, tuple)) else [modules]
        self._clip_norm = clip_norm
        self._weight_decay = weight_decay
        self._l2_reg = l2_reg
        self._wdpattern = wdpattern
        if scales is not None:
            assert isinstance(scales, (list, tuple)), scales
            assert len(scales) == len(self._modules), (len(scales), len(self._modules))
        self._scales = scales
        self._opt = select_optimizer(opt_name)(lr, **kwargs)
        self._return_grads = return_grads
        # useful for mixed precision training on GPUs to
        # avoid numerical underflow caused by using float16 gradients
        prec_policy = prec.global_policy()
        self._mpt = prec_policy.compute_dtype != prec_policy.variable_dtype
        if self._mpt:
            do_logging(
                'Mixed precision training will be performed', 
                logger=logger)
            self._opt = prec.LossScaleOptimizer(self._opt)
        # we do not initialize variables here as modules may not be initialized at this point
        self._variables = None

    def get_weights(self):
        return self._opt.get_weights()
    
    def set_weights(self, weights):
        self._opt.set_weights(weights)

    @property
    def variables(self):
        return self._opt.variables()
    
    def get_transformed_grads(self, var_list=[]):
        assert hasattr(self._opt, 'get_transformed_grads'), f'{self._opt} does not support "get_transformed_grads"'
        return self._opt.get_transformed_grads(var_list or self._variables)

    def __call__(self, tape=None, loss=None, grads=None, output_gradients=None):
        if loss is None and grads is None:
            raise ValueError('Neither loss nor grads is provided')
        if loss is not None and grads is not None:
            raise ValueError('Both loss and grads are provvided')
        if isinstance(loss, tf.Tensor) and loss.shape != ():
            raise ValueError(f'loss is expected to be a scalar Tensor, but get {loss}')

        if self._variables is None:
            variables = [m.trainable_variables for m in self._modules]
            for v, m in zip(variables, self._modules):
                do_logging(f'Found {len(v)} parameters for {m.name}', logger=logger)
            self._variables = tf.nest.flatten(variables)
            if self._scales is not None:
                scales = [[self._scales[i] for _ in m.trainable_variables] 
                    for i, m in enumerate(self._modules)]
                self._scales = tf.nest.flatten(scales)

        if grads is None:
            if tape is None:
                raise ValueError('tf.GradientTape is ')
            if self._l2_reg:
                loss = self._add_l2_regularization(loss)
            if self._mpt:
                with tape:
                    loss = self._opt.get_scaled_loss(loss)
            grads = tape.gradient(loss, self._variables, output_gradients=output_gradients)
        if None in grads:
            raise ValueError(f'No grads for {self._variables[grads.index(None)].name}')
        if self._mpt:
            grads = self._opt.get_unscaled_gradients(grads)
        if self._scales is not None:
            assert len(grads) == len(self._scales), (len(grads), len(self._scales))
            grads = [g * s for g, s in zip(grads, self._scales)]
        norm = tf.linalg.global_norm(grads)
        if self._clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm, norm)
        if self._weight_decay:
            self._apply_weight_decay()
        self.grads = grads
        self._opt.apply_gradients(zip(grads, self._variables))

        if self._return_grads:
            return norm, {v.name: g for v, g in zip(self._variables, grads)}
        else:
            return norm
    
    def _add_l2_regularization(self, loss):
        do_logging(f'Apply L2 regularization with coefficient: {self._l2_reg}\n" \
            "Wait, are you sure you want to apply l2 regularization instead of weight decay?',
            logger=logger)
        for var in self._variables:
            loss += self._l2_reg * tf.nn.l2_loss(var)
        return loss

    def _apply_weight_decay(self):
        do_logging(f'Apply weight decay with coefficient: {self._weight_decay}',
            logger=logger)
        for var in self._variables:
            if re.search(self._wdpattern, var.name):
                print(var.name, self._weight_decay)
                var.assign((1 - self._weight_decay) * var)

if __name__ == '__main__':
    l = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(.01))
    tf.random.set_seed(0)
    opt = Optimizer('adam', l, 1, weight_decay=.1)
    x = tf.random.normal((32, 2))
    with tf.GradientTape() as t:
        y = l(x)
        loss = tf.reduce_mean((y - 1)**2)
    opt(t, loss)
    print(l.variables)
    