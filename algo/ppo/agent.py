import numpy as np
import tensorflow as tf

from utility.display import pwc
from core.tf_config import build
from core.base import BaseAgent
from core.decorator import agent_config
from algo.ppo.loss import compute_ppo_loss, compute_value_loss


class Agent(BaseAgent):
    @agent_config
    def __init__(self, 
                name, 
                config, 
                models, 
                state_shape,
                state_dtype,
                action_dim,
                action_dtype,
                n_envs):
        # optimizer
        if self.optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                epsilon=self.epsilon
            )
        elif self.optimizer.lower() == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate,
                rho=.99,
                epsilon=self.epsilon
            )
        else:
            raise NotImplementedError()

        self.ckpt_models['optimizer'] = self.optimizer

        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = [
            (state_shape, state_dtype, 'state'),
            ([action_dim], action_dtype, 'action'),
            ([1], tf.float32, 'traj_ret'),
            ([1], tf.float32, 'value'),
            ([1], tf.float32, 'advantage'),
            ([1], tf.float32, 'old_logpi'),
            ([1], tf.float32, 'mask'),
            ((), tf.float32, 'n'),
        ]
        self.train_step = build(
            self._train_step, 
            TensorSpecs, 
            sequential=True, 
            batch_size=n_envs
        )

    def train_log(self, buffer, early_terminate, epoch):
        for i in range(self.n_updates):
            self.ac.reset_states()
            for j in range(self.n_minibatches):
                data = buffer.get_batch()
                data['n'] = n = np.sum(data['mask'])
                value = data['value']
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                with tf.name_scope('train'):
                    loss_info  = self.train_step(**data)
                    entropy, approx_kl, p_clip_frac, v_clip_frac, ppo_loss, value_loss, global_norm = loss_info
    
                n_total_trans = value.size
                n_valid_trans = n or n_total_trans
                self.store(
                    ppo_loss=ppo_loss.numpy(), 
                    value_loss=value_loss.numpy(),
                    entropy=entropy.numpy(), 
                    p_clip_frac=p_clip_frac.numpy(),
                    v_clip_frac=v_clip_frac.numpy(),
                    value=np.mean(value),
                    global_norm=global_norm.numpy(),
                    n_valid_trans=n_valid_trans,
                    n_total_trans=n_total_trans,
                    valid_trans_frac = n_valid_trans / n_total_trans
                )
            
            if self.max_kl and early_terminate and approx_kl > self.max_kl:
                pwc(f'Eearly stopping at epoch-{epoch} update-{i+1} due to reaching max kl.',
                    f'Current kl={approx_kl:.3g}', color='blue')
                break
        self.store(approx_kl=approx_kl)

    @tf.function
    def _train_step(self, state, action, traj_ret, value, advantage, old_logpi, mask=None, n=None):
        with tf.GradientTape() as tape:
            old_value = value
            logpi, entropy, value = self.ac.train_step(state, action)
            # policy loss
            ppo_loss, entropy, approx_kl, p_clip_frac = compute_ppo_loss(
                logpi, old_logpi, advantage, self.clip_range,
                entropy, mask=mask, n=n)
            # value loss
            value_loss, v_clip_frac = compute_value_loss(
                value, traj_ret, old_value, self.clip_range,
                mask=mask, n=n)

            with tf.name_scope('total_loss'):
                policy_loss = (ppo_loss 
                        - self.entropy_coef * entropy # TODO: adaptive entropy regularizer
                        + self.kl_coef * approx_kl)
                value_loss = self.value_coef * value_loss
                total_loss = policy_loss + value_loss

        with tf.name_scope('gradient'):
            grads = tape.gradient(total_loss, self.ac.trainable_variables)
            if hasattr(self, 'clip_norm'):
                grads, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.optimizer.apply_gradients(zip(grads, self.ac.trainable_variables))

        return entropy, approx_kl, p_clip_frac, v_clip_frac, ppo_loss, value_loss, global_norm 
