import numpy as np
import tensorflow as tf


class TempLearner:
    def _learn_temp(self, x, entropy, IS_ratio):
        terms = {}
        if self.temperature.is_trainable():
            # Entropy of a uniform distribution
            self._target_entropy = np.log(self._action_dim)
            target_entropy_coef = self._target_entropy_coef \
                if isinstance(self._target_entropy_coef, float) \
                else self._target_entropy_coef(self._train_step)
            target_entropy = self._target_entropy * target_entropy_coef
            with tf.GradientTape() as tape:
                log_temp, temp = self.temperature(x)
                entropy_diff = entropy - target_entropy
                temp_loss = log_temp * entropy_diff
                tf.debugging.assert_shapes([[temp_loss, (None, )]])
                temp_loss = tf.reduce_mean(IS_ratio * temp_loss)
            terms['target_entropy'] = target_entropy
            terms['entropy_diff'] = entropy_diff
            terms['log_temp'] = log_temp
            terms['temp_loss'] = temp_loss
            terms['temp_norm'] = self._temp_opt(tape, temp_loss)

        return terms
    
class DiscreteRegularizer:
    def _add_regularizer_attr(self):
        self._regularizer = getattr(self, '_regularizer', 'entropy').lower()
        if self._regularizer == 'tsallis':
            self._tsallis_q = getattr(self, '_tsallis_q', 1.2)

    def _compute_regularization(self, pi):
        if self._regularizer is None:
            return 1
        elif self._regularizer == 'entropy':
            phi = tf.math.log(tf.maximum(pi, 1e-8))
        elif self._regularizer == 'tsallis':
            phi = (pi**(self._tsallis_q - 1) - 1) / (self._tsallis_q - 1)
        else:
            raise NotImplementedError
        
        return -tf.reduce_sum(pi * phi, axis=-1)