import logging
import numpy as np
import tensorflow as tf

from utility.rl_loss import ppo_loss
from core.tf_config import build
from core.decorator import override
from core.decorator import step_track
from core.optimizer import Optimizer
from algo.ppo.agent import Agent as PPOAgent
from algo.rnd.rnd import RND


logger = logging.getLogger(__name__)

class Agent(PPOAgent):
    @override(PPOAgent)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._n_envs = env.n_envs
        self.eval_reward_int = []
        self.eval_reward_ext = []
        self.eval_actions = []
        rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'
        self.rnd = RND(self.model, self._gamma_int, rms_path)

    @override(PPOAgent)
    def _construct_optimizers(self):
        # ac = [self.encoder, self.actor, self.value]
        ac = self.ac
        self._ac_opt = Optimizer(
            self._optimizer, ac, self._ac_lr, 
            clip_norm=self._clip_norm)

        # optimizer
        self._pred_opt = Optimizer(
            self._optimizer, self.predictor, self._pred_lr
        )

    @override(PPOAgent)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        norm_obs_shape = env.obs_shape[:-1] + (1,)
        TensorSpecs = dict(
            obs=(env.obs_shape, env.obs_dtype, 'obs'),
            norm_obs=(norm_obs_shape, tf.float32, 'norm_obs'),
            action=(env.action_shape, env.action_dtype, 'action'),
            traj_ret_int=((), tf.float32, 'traj_ret_int'),
            traj_ret_ext=((), tf.float32, 'traj_ret_ext'),
            value_int=((), tf.float32, 'value_int'),
            value_ext=((), tf.float32, 'value_ext'),
            advantage=((), tf.float32, 'advantage'),
            logpi=((), tf.float32, 'logpi'),
        )
        self.learn = build(self._learn, TensorSpecs)

    @tf.function
    def _summary(self, data, terms):
        tf.summary.histogram('sum/value_int', data['value_int'], step=self._env_step)
        tf.summary.histogram('sum/value_ext', data['value_ext'], step=self._env_step)

    """ Call """
    @override(PPOAgent)
    def _process_input(self, obs, evaluation, env_output):
        # update rms and normalize obs
        if evaluation:
            norm_obs = np.expand_dims(obs, 1)
            norm_obs = self.normalize_obs(norm_obs)
            reward_int = self.compute_int_reward(norm_obs)
            self.eval_reward_int.append(np.squeeze(reward_int))
            self.eval_reward_ext.append(np.squeeze(env_output.reward))
        return obs, {}

    @override(PPOAgent)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = super()._process_output(obs, kwargs, out, evaluation)
        if evaluation:
            assert not isinstance(out, (list, tuple)), out
            self.eval_actions.append(out)
        return out

    @override(PPOAgent)
    def record_last_env_output(self, env_output):
        self._last_obs = env_output.obs

    @override(PPOAgent)
    def compute_value(self, obs=None):
        obs = obs or self._last_obs
        out = self.model.compute_value(obs)
        return tf.nest.map_structure(lambda x: x.numpy(), out)

    @step_track
    def learn_log(self, step):
        for i in range(self.N_UPDATES):
            for j in range(1, self.N_MBS+1):
                data = self.dataset.sample()
                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                terms = self.learn(**data)

                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                kl = terms.pop('train/kl')
                value_int = terms.pop('train/value_int')
                value_ext = terms.pop('train/value_ext')
                self.store(
                    **terms, 
                    **{
                        'train/value_int': np.mean(value_int),
                        'train/value_int_max': np.max(value_int),
                        'train/value_ext': np.mean(value_ext),
                        'train/value_ext_max': np.max(value_ext),
                    })
                if getattr(self, '_max_kl', None) and kl > self._max_kl:
                    break
                if self._value_update == 'reuse':
                    self.dataset.update('value_int', value_int)
                    self.dataset.update('value_ext', value_ext)
            if getattr(self, '_max_kl', None) and kl > self._max_kl:
                logger.info(f'{self._model_name}: Eearly stopping after '
                    f'{i*self.N_MBS+j} update(s) due to reaching max kl.'
                    f'Current kl={kl:.3g}')
                break
            
            if self._value_update == 'once':
                self.dataset.update_value_with_func(self.compute_value)
            if self._value_update is not None:
                last_value = self.compute_value()
                self.dataset.finish(last_value)
        self.store(**{'train/kl': kl})

        if self._to_summary(step):
            self._summary(data, terms)

        return i * self.N_MBS + j

    @tf.function
    def _learn(self, obs, norm_obs, action, traj_ret_int, traj_ret_ext, 
            value_int, value_ext, advantage, logpi):
        old_value_int, old_value_ext = value_int, value_ext
        norm_obs = tf.reshape(norm_obs, 
            (self._n_envs, self.N_STEPS // self.N_MBS, *norm_obs.shape[-3:]))
        terms = {}
        with tf.GradientTape() as pred_tape:
            target_feat = tf.stop_gradient(self.target(norm_obs))
            pred_feat = self.predictor(norm_obs)
            pred_loss = tf.reduce_mean(tf.square(target_feat - pred_feat), axis=-1)
            tf.debugging.assert_shapes([[pred_loss, (self._n_envs, self.N_STEPS // self.N_MBS)]])
            mask = tf.random.uniform(pred_loss.shape, maxval=1., dtype=pred_loss.dtype)
            mask = tf.cast(mask < self._pred_frac, pred_loss.dtype)
            pred_loss = tf.reduce_sum(mask * pred_loss) / tf.maximum(tf.reduce_sum(mask), 1)
        terms['pred_norm'] = self._pred_opt(pred_tape, pred_loss)

        with tf.GradientTape() as ac_tape:
            act_dist, value_int, value_ext = self.ac(obs, return_value=True)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            # policy loss
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy)
            # value loss
            value_loss_int, v_int_clip_frac = self._compute_value_loss(value_int, traj_ret_int, old_value_int)
            value_loss_ext, v_out_clip_frac = self._compute_value_loss(value_ext, traj_ret_ext, old_value_ext)

            entropy_loss = - self._entropy_coef * entropy
            actor_loss = policy_loss + entropy_loss
            v_loss_int = self._v_coef * value_loss_int
            v_loss_ext = self._v_coef * value_loss_ext
            ac_loss = actor_loss + value_loss_int + value_loss_ext
        terms['ac_norm'] = self._ac_opt(ac_tape, ac_loss)

        target_feat_mean, target_feat_var = tf.nn.moments(target_feat, axes=[0, 1])
        pred_feat_mean, pred_feat_var = tf.nn.moments(pred_feat, axes=[0, 1])
        terms.update(dict(
            target_feat_mean=target_feat_mean,
            target_feat_var=target_feat_var,
            target_feat_max=tf.reduce_max(tf.abs(target_feat)),
            pred_feat_mean=pred_feat_mean,
            pred_feat_var=pred_feat_var,
            pred_feat_max=tf.reduce_max(tf.abs(pred_feat)),
            pred_loss=pred_loss,
            advantage=advantage, 
            ratio=tf.exp(log_ratio), 
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            entropy_loss=entropy_loss,
            value_int=value_int,
            value_ext=value_ext,
            v_loss_int=v_loss_int,
            v_loss_ext=v_loss_ext,
            v_int_clip_frac=v_int_clip_frac,
            v_out_clip_frac=v_out_clip_frac,
        ))

        return terms

    """ RND Methods """
    def compute_int_reward(self, next_obs):
        return self.rnd.compute_int_reward(next_obs)

    def update_obs_rms(self, obs):
        self.rnd.update_obs_rms(obs)
    
    def normalize_obs(self, obs):
        return self.rnd.normalize_obs(obs)
    
    def restore(self):
        super().restore()
        self.rnd.restore()
    
    def save(self, print_terminal_info=False):
        super().save(print_terminal_info=print_terminal_info)
        self.rnd.save()
    
    def rnd_rms_restored(self):
        return self.rnd.rms_restored()

    """ Evaluation Methods """
    def retrieve_eval_rewards(self):
        reward_int = np.array(self.eval_reward_int)
        reward_ext = np.array(self.eval_reward_ext)
        self.eval_reward_int.clear()
        self.eval_reward_ext.clear()
        return reward_int, reward_ext

    def retrieve_eval_actions(self):
        action = np.array(self.eval_actions)
        self.eval_actions.clear()
        return action

    @override(PPOAgent)
    def get_running_stats(self):
        stats = ()
        if self._normalize_obs:
            stats += self._obs_rms.get_stats()
        if self._normalize_reward:
            stats += self._reward_rms.get_stats()
        assert stats == (), stats
        stats = self.rnd.get_running_stats()
        return stats