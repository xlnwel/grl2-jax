import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops


class RMSprop(optimizer_v2.OptimizerV2):
    r"""Optimizer that implements the RMSprop algorithm.
    A detailed description of rmsprop.
        - maintain a moving (discounted) average of the square of gradients
        - divide gradient by the root of this average
    $$mean_square_t = rho * mean_square{t-1} + (1-rho) * gradient ** 2$$
    $$mom_t = momentum * mom_{t-1} + learning_rate * gradient / \sqrt{ /
        mean_square_t + \epsilon}$$
    $$variable_t := variable_{t-1} - mom_t$$
    This implementation of RMSprop uses plain momentum, not Nesterov momentum.
    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance:
    $$mean_grad_t = rho * mean_grad_{t-1} + (1-rho) * gradient$$
    $$mean_square_t = rho * mean_square_{t-1} + (1-rho) * gradient ** 2$$
    $$mom_t = momentum * mom_{t-1} + learning_rate * gradient /
        sqrt(mean_square_t - mean_grad_t**2 + epsilon)$$
    $$variable_t := variable_{t-1} - mom_t$$
    References
        See ([pdf]
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
    """

    def __init__(self,
                learning_rate=0.001,
                rho=0.9,
                momentum=0.0,
                epsilon=1e-7,
                centered=False,
                name="RMSprop",
                **kwargs):
        """Construct a new RMSprop optimizer.
        Note that in the dense implementation of this algorithm, variables and their
        corresponding accumulators (momentum, gradient moving average, square
        gradient moving average) will be updated even if the gradient is zero
        (i.e. accumulators will decay, momentum will be applied). The sparse
        implementation (used when the gradient is an `IndexedSlices` object,
        typically because of `tf.gather` or an embedding lookup in the forward pass)
        will not update variable slices or their accumulators unless those slices
        were used in the forward pass (nor is there an "eventual" correction to
        account for these omitted updates). This leads to more efficient updates for
        large embedding lookup tables (where most of the slices are not accessed in
        a particular graph execution), but differs from the published algorithm.
        Args:
        learning_rate: A Tensor or a floating point value.  The learning rate.
        rho: Discounting factor for the history/coming gradient
        momentum: A scalar tensor.
        epsilon: Small value to avoid zero denominator.
        centered: If True, gradients are normalized by the estimated variance of
            the gradient; if False, by the uncentered second moment. Setting this to
            True may help with training, but is slightly more expensive in terms of
            computation and memory. Defaults to False.
        name: Optional name prefix for the operations created when applying
            gradients. Defaults to "RMSprop".  @compatibility(eager) When eager
            execution is enabled, `learning_rate`, `decay`, `momentum`, and
            `epsilon` can each be a callable that takes no arguments and returns the
            actual value to use. This can be useful for changing these values across
            different invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        """
        super(RMSprop, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("rho", rho)

        self._momentum = False
        if momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.epsilon = epsilon
        self.centered = centered
        self._var_grad_map = {}

    def get_transformed_grads(self, variables=None):
        if variables: 
            return [self._var_grad_map[v.name] for v in variables]
        else:
            return list(self._var_grad_map.values())

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.centered:
            for var in var_list:
                self.add_slot(var, "mg")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(RMSprop, self)._prepare_local(var_device, var_dtype, apply_state)

        rho = array_ops.identity(self._get_hyper("rho", var_dtype))
        apply_state[(var_device, var_dtype)].update(dict(
            neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
            rho=rho,
            momentum=array_ops.identity(self._get_hyper("momentum", var_dtype)),
            one_minus_rho=1. - rho
        ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return training_ops.resource_apply_centered_rms_prop(
                    var.handle,
                    mg.handle,
                    rms.handle,
                    mom.handle,
                    coefficients["lr_t"],
                    coefficients["rho"],
                    coefficients["momentum"],
                    coefficients["epsilon"],
                    grad,
                    use_locking=self._use_locking)
            else:
                return training_ops.resource_apply_rms_prop(
                    var.handle,
                    rms.handle,
                    mom.handle,
                    coefficients["lr_t"],
                    coefficients["rho"],
                    coefficients["momentum"],
                    coefficients["epsilon"],
                    grad,
                    use_locking=self._use_locking)
        else:
            rms_t = (coefficients["rho"] * rms +
                coefficients["one_minus_rho"] * math_ops.square(grad))
            rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
            denom_t = rms_t
            if self.centered:
                mg = self.get_slot(var, "mg")
                mg_t = coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
                mg_t = state_ops.assign(mg, mg_t, use_locking=self._use_locking)
                denom_t = rms_t - math_ops.square(mg_t)
            coef = - coefficients["lr_t"] / (
                math_ops.sqrt(denom_t) + coefficients["epsilon"])
            self._var_grad_map[var.name] = transformed_grads = coef * grad
            
            op = state_ops.assign_add(var, transformed_grads, use_locking=self._use_locking).op
            return op

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(RMSprop, self).set_weights(weights)

    def get_config(self):
        config = super(RMSprop, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "rho": self._serialize_hyperparameter("rho"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "epsilon": self.epsilon,
            "centered": self.centered,
        })
        return 