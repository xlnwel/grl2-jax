import numpy as np
import tensorflow as tf

from utility.schedule import TFPiecewiseSchedule
from utility.tf_utils import explained_variance, tensor2numpy
from utility.rl_loss import ppo_loss
from core.mixin import Memory
from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase


def infer_life_mask(mask, discount, concat=True):
    life_mask = np.logical_or(
        mask, 1-np.any(discount, 1, keepdims=True)).astype(np.float32)
    # np.testing.assert_equal(life_mask, mask)
    if concat:
        life_mask = np.concatenate(life_mask)
    return life_mask

def collect(buffer, env, step, reset, life_mask, reward, 
            discount, next_obs, **kwargs):
    kwargs['life_mask'] = infer_life_mask(discount, discount)
    kwargs['reward'] = np.concatenate(reward)
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = np.concatenate(discount)
    buffer.add(**kwargs)

def random_actor(env_output, env=None, **kwargs):
    obs = env_output.obs
    a = np.concatenate(env.random_action())
    terms = {
        'obs': np.concatenate(obs['obs']), 
        'global_state': np.concatenate(obs['global_state']),
        'life_mask': obs['life_mask'],
    }
    return a, terms

class Agent(Memory, PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self._setup_memory_state_record()

        self._n_agents = env.n_agents
        self._value_life_mask = self._value_life_mask or None
        self._actor_life_mask = self._actor_life_mask or None
        
        state_keys = self.model.state_keys
        mid = len(state_keys) // 2
        self._actor_state_keys = state_keys[:mid]
        self._value_state_keys = state_keys[mid:]
        self._value_sample_keys = [
            'global_state', 'value', 
            'traj_ret', 'life_mask', 'mask'
        ] + list(self._value_state_keys)

        self._basic_shape = (self._sample_size, )

    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list), self._lr
            self._lr = TFPiecewiseSchedule(self._lr)

        actor_models = [v for k, v in self.model.items() if 'actor' in k]
        self._actor_opt = self._construct_opt(actor_models, self._lr, 
            weight_decay=self._actor_weight_decay)
        value_models = [v for k, v in self.model.items() if 'value' in k]
        self._value_opt = self._construct_opt(value_models, self._lr)

        return actor_models + value_models
    
    """ PPO methods """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((*self._basic_shape, *env.obs_shape), env.obs_dtype, 'obs'),
            global_state=((*self._basic_shape, *env.shared_state_shape), env.shared_state_dtype, 'global_state'),
            action_mask=((*self._basic_shape, env.action_dim), tf.bool, 'action_mask'),
            action=((*self._basic_shape, *env.action_shape), env.action_dtype, 'action'),
            value=(self._basic_shape, tf.float32, 'value'),
            traj_ret=(self._basic_shape, tf.float32, 'traj_ret'),
            advantage=(self._basic_shape, tf.float32, 'advantage'),
            logpi=(self._basic_shape, tf.float32, 'logpi'),
            life_mask=(self._basic_shape, tf.float32, 'life_mask'),
            mask=(self._basic_shape, tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = (
                    (*self._basic_shape, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = (
                    self._basic_shape, self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs)

        TensorSpecs = dict(
            global_state=((*self._basic_shape, *env.shared_state_shape), env.shared_state_dtype, 'global_state'),
            value=(self._basic_shape, tf.float32, 'value'),
            traj_ret=(self._basic_shape, tf.float32, 'traj_ret'),
            life_mask=(self._basic_shape, tf.float32, 'life_mask'),
            mask=(self._basic_shape, tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.value_state_size)
            TensorSpecs['value_state'] = state_type(*[((sz, ), self._dtype, f'value_{name}') 
                for name, sz in self.model.value_state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = (
                    (self._sample_size, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = (
                    (self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        
        learn_value = tf.function(self._learn_value)
        self.learn_value = build(learn_value, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    """ Call """
    def _reshape_env_output(self, env_output):
        """ merges the batch and agent dimensions """
        obs, reward, discount, reset = env_output
        new_obs = {}
        for k, v in obs.items():
            new_obs[k] = np.concatenate(v)
        # reward and discount are not used for inference so we do not process them
        # reward = np.concatenate(reward)
        # discount = np.concatenate(discount)
        reset = np.concatenate(reset)
        return type(env_output)(new_obs, reward, discount, reset)

    # @override(PPOBase)
    def _process_input(self, env_output, evaluation):
        if evaluation:
            self._process_obs(env_output.obs, update_rms=False)
        else:
            life_mask = env_output.obs['life_mask']
            self._process_obs(env_output.obs, mask=life_mask)
        mask = self._get_mask(env_output.reset)
        obs, kwargs = self._divide_obs(env_output.obs)
        kwargs = self._add_memory_state_to_kwargs(
            obs, mask=mask, kwargs=kwargs)
        return obs, kwargs

    # @override(PPOBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        out = tensor2numpy(out)
        out = self._add_non_tensors_to_terms(obs, kwargs, out, evaluation)
        return out
    
    def _add_non_tensors_to_terms(self, obs, kwargs, out, evaluation):
        if evaluation:
            out = [out]
        else:
            out[1].update({
                'obs': obs, # ensure obs is placed in terms even when no observation normalization is performed
                'global_state': kwargs['global_state'],
                'mask': kwargs['mask'],
                'action_mask': kwargs['action_mask'],
                'life_mask': kwargs['life_mask'].reshape(-1, self._n_agents),
            })
        return out

    """ PPO methods """
    # @override(PPOBase)
    def record_last_env_output(self, env_output):
        self._env_output = self._reshape_env_output(env_output)
        self._process_obs(self._env_output.obs, update_rms=False)
        mask = self._get_mask(self._env_output.reset)
        self._state = self._apply_mask_to_state(self._state, mask)

    # @override(PPOBase)
    def compute_value(self, global_state=None, state=None, mask=None, prev_reward=None, return_state=False):
        # be sure obs is normalized if obs normalization is required
        if global_state is None:
            global_state = self._env_output.obs['global_state']
        if state is None:
            state = self._state
        mid = len(self._state) // 2
        state = state[mid:]
        if mask is None:
            mask = self._get_mask(self._env_output.reset)
        kwargs = dict(
            state=state,
            mask=mask,
            return_state=return_state,
        )
        out = self.model.compute_value(global_state, **kwargs)
        return tensor2numpy(out)

    @tf.function
    def _learn(self, obs, global_state, action_mask, action, value, 
            traj_ret, advantage, logpi, state=None, life_mask=None, 
            mask=None, prev_action=None, prev_reward=None):
        actor_state, value_state = self.model.split_state(state)

        actor_terms = self._learn_actor(
            obs, action_mask, action, advantage, logpi, 
            actor_state, life_mask, mask, prev_action, prev_reward)
        value_terms = self._learn_value(global_state, value, traj_ret, 
            value_state, life_mask, mask, prev_action, prev_reward)

        terms = {**actor_terms, **value_terms}

        return terms

    def _learn_actor(self, obs, action_mask, action, advantage, logpi, 
                    actor_state=None, life_mask=None, mask=None, 
                    prev_action=None, prev_reward=None):
        with tf.GradientTape() as tape:
            x_actor, _ = self.model.encode(
                obs, actor_state, mask, 'actor',
                prev_action=prev_action, prev_reward=prev_reward)
            act_dist = self.actor(x_actor, action_mask)
            new_logpi = act_dist.log_prob(action)
            entropy = act_dist.entropy()
            log_ratio = new_logpi - logpi
            policy_loss, entropy, kl, p_clip_frac = ppo_loss(
                log_ratio, advantage, self._clip_range, entropy, 
                self._actor_life_mask and life_mask)
            actor_loss = policy_loss - self._entropy_coef * entropy
        
        actor_norm = self._actor_opt(tape, actor_loss)
        n_actions = tf.reduce_sum(tf.cast(action_mask, tf.float32), -1)
        uniform_prob = 1 / n_actions
        terms = dict(
            actor_norm=actor_norm,
            n_avail_actions=n_actions,
            uniform_entropy=-tf.reduce_sum(
                tf.math.log(uniform_prob) * life_mask) / tf.reduce_sum(life_mask),
            entropy=entropy,
            kl=kl,
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
        )

        return terms

    def _learn_value(self, global_state, value, traj_ret, 
                    value_state=None, life_mask=None, mask=None, 
                    prev_action=None, prev_reward=None):
        old_value = value
        with tf.GradientTape() as tape:
            x_value, _ = self.model.encode(
                global_state, value_state, mask, 'value',
                prev_action=prev_action, prev_reward=prev_reward)
            value = self.value(x_value)

            value_loss, v_clip_frac = self._compute_value_loss(
                value, traj_ret, old_value,
                self._value_life_mask and life_mask)
            value_loss = self._value_coef * value_loss
        
        value_norm = self._value_opt(tape, value_loss)
        terms = dict(
            value=value,
            value_norm=value_norm,
            v_loss=value_loss,
            explained_variance=explained_variance(traj_ret, value),
            v_clip_frac=v_clip_frac,
        )

        return terms

    def _divide_obs(self, obs):
        kwargs = {
            'global_state': obs['global_state'],
            'action_mask': obs['action_mask'].astype(np.bool),
            'life_mask': obs['life_mask']
        }
        obs = obs['obs']
        return obs, kwargs

    def _sample_learn(self):
        n = super()._sample_learn()

        for _ in range(self.N_VALUE_EPOCHS):
            for _ in range(self.N_MBS):
                data = self.dataset.sample(self._value_sample_keys)

                data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
                
                terms = self.learn_value(**data)
                terms = {f'train/{k}': v.numpy() for k, v in terms.items()}
                self.store(**terms)
        
        return n

    def _store_buffer_stats(self):
        self.store(**self.dataset.compute_mean_max_std('reward'))
        # self.store(**self.dataset.compute_mean_max_std('obs'))
        # self.store(**self.dataset.compute_mean_max_std('global_state'))
        self.store(**self.dataset.compute_mean_max_std('advantage'))
        self.store(**self.dataset.compute_mean_max_std('value'))
        self.store(**self.dataset.compute_mean_max_std('traj_ret'))
        self.store(**self.dataset.compute_fraction('life_mask'))
        self.store(**self.dataset.compute_fraction('action_mask'))
        self.store(**self.dataset.compute_fraction('mask'))
        self.store(**self.dataset.compute_fraction('discount'))
