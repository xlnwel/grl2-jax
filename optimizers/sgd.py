import numpy as np

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import state_ops


class SGD(optimizer_v2.OptimizerV2):
    def __init__(self,
                learning_rate=0.001,
                name="SGD",
                **kwargs):
        super().__init__(name)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        self._var_grad_map = {}
        self._grad_coef_map = {}

    def get_transformed_grads(self, variables=None):
        if variables: 
            return {v.name: self._var_grad_map[v.name] for v in variables}
        else:
            return self._var_grad_map

    def get_grad_coefs(self, variables=None):
        if variables: 
            return [self._grad_coef_map[v.name] for v in variables]
        else:
            return list(self._grad_coef_map.values())

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        self._grad_coef_map[var.name] = - coefficients["lr_t"]
        self._var_grad_map[var.name] = transformed_grads = self._grad_coef_map[var.name] * grad
        
        op = state_ops.assign_add(var, transformed_grads, use_locking=self._use_locking).op
        return op

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras V1 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        })
        return 