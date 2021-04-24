import numpy as np
import tensorflow as tf

from utility.schedule import TFPiecewiseSchedule
from utility.tf_utils import explained_variance
from utility.rl_loss import ppo_loss
from core.base import Memory
from core.tf_config import build
from core.decorator import override
from algo.ppo.base import PPOBase

def collect(buffer, env, step, reset, reward, discount, next_obs, **kwargs):
    kwargs['reward'] = np.concatenate(reward)
    kwargs['life_mask'] = np.concatenate(discount)
    # discount is zero only when all agents are done
    discount[np.any(discount, 1)] = 1
    kwargs['discount'] = np.concatenate(discount)
    buffer.add(**kwargs)

class Agent(Memory, PPOBase):
    """ Initialization """
    @override(PPOBase)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)
        self.n_agents = env.n_agents
        self._setup_memory_state_record()

    @override(PPOBase)
    def _construct_optimizers(self):
        if getattr(self, 'schedule_lr', False):
            assert isinstance(self._lr, list), self._lr
            self._lr = TFPiecewiseSchedule(self._lr)
        actor_models = [v for k, v in self.model.items() if 'actor' in k]
        self._actor_opt = self._construct_opt(actor_models, self._lr)
        value_models = [v for k, v in self.model.items() if 'value' in k]
        self._value_opt = self._construct_opt(value_models, self._lr)
    
    """ PPO methods """
    @override(PPOBase)
    def _build_learn(self, env):
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = dict(
            obs=((self._sample_size, *env.obs_shape), env.obs_dtype, 'obs'),
            shared_state=((self._sample_size, *env.shared_state_shape), env.shared_state_dtype, 'shared_state'),
            action_mask=((self._sample_size, env.action_dim), tf.bool, 'action_mask'),
            action=((self._sample_size, *env.action_shape), env.action_dtype, 'action'),
            value=((self._sample_size, ), tf.float32, 'value'),
            traj_ret=((self._sample_size, ), tf.float32, 'traj_ret'),
            advantage=((self._sample_size, ), tf.float32, 'advantage'),
            logpi=((self._sample_size, ), tf.float32, 'logpi'),
            life_mask=((self._sample_size,), tf.float32, 'life_mask'),
            mask=((self._sample_size,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['state'] = state_type(*[((sz, ), self._dtype, name) 
                for name, sz in self.model.state_size._asdict().items()])
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = ((self._sample_size, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = ((self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        self.learn = build(self._learn, TensorSpecs)

        TensorSpecs = dict(
            shared_state=((self._sample_size, *env.shared_state_shape), env.shared_state_dtype, 'shared_state'),
            value=((self._sample_size,), tf.float32, 'value'),
            traj_ret=((self._sample_size,), tf.float32, 'traj_ret'),
            life_mask=((self._sample_size,), tf.float32, 'life_mask'),
            mask=((self._sample_size,), tf.float32, 'mask'),
        )
        if self._store_state:
            state_type = type(self.model.state_size)
            TensorSpecs['value_state'] = (
                (self.model.state_size.value_h, ), self._dtype, 'value_state')
        if self._additional_rnn_inputs:
            if 'prev_action' in self._additional_rnn_inputs:
                TensorSpecs['prev_action'] = ((self._sample_size, *env.action_shape), env.action_dtype, 'prev_action')
            if 'prev_reward' in self._additional_rnn_inputs:
                TensorSpecs['prev_reward'] = ((self._sample_size,), self._dtype, 'prev_reward')    # this reward should be unnormlaized
        
        learn_value = tf.function(self._learn_value)
        self.learn_value = build(learn_value, TensorSpecs)

    # @override(PPOBase)
    # def _summary(self, data, terms):
    #     tf.summary.histogram('sum/value', data['value'], step=self._env_step)
    #     tf.summary.histogram('sum/logpi', data['logpi'], step=self._env_step)

    """ Call """
    def _reshape_output(self, env_output):
        """ merge the batch and agent dimensions """
        obs, reward, discount, reset = env_output
        new_obs = {}
        for k, v in obs.items():
            new_obs[k] = np.concatenate(v)
            assert new_obs[k].ndim ==2, new_obs[k].shape    # don't consider images
        discount = np.concatenate(discount)
        reset = np.concatenate(reset)
        return type(env_output)(new_obs, reward, discount, reset)

    # @override(PPOBase)
    def _process_input(self, env_output, evaluation):
        if evaluation:
            self._process_obs(env_output.obs, update_rms=False)
        else:
            life_mask = env_output.discount
            self._process_obs(env_output.obs, mask=life_mask)
        mask = 1. - env_output.reset
        obs, kwargs = self._divide_obs(env_output.obs)
        obs, kwargs = self._add_memory_state_to_kwargs(obs, mask=mask, kwargs=kwargs)
        return obs, kwargs

    # @override(PPOBase)
    def _process_output(self, obs, kwargs, out, evaluation):
        out = self._add_tensors_to_terms(obs, kwargs, out, evaluation)
        out = super()._process_output(obs, kwargs, out, evaluation)
        out = self._add_non_tensors_to_terms(obs, out, kwargs, evaluation)
        return out
    
    def _add_non_tensors_to_terms(self, obs, out, kwargs, evaluation):
        if evaluation:
            out = [out]
        else:
            out[1].update({
                'obs': obs, # ensure obs is placed in terms even when no observation normalization is performed
                'mask': kwargs['mask'],
                'action_mask': kwargs['action_mask'],
                'shared_state': kwargs['shared_state'],
            })
        return out

    """ PPO methods """
    @override(PPOBase)
    def record_last_env_output(self, env_output):
        self._env_output = self._reshape_output(env_output)
        self._process_obs(self._env_output.obs, update_rms=False)    

    @override(PPOBase)
    def compute_value(self, shared_state=None, state=None, mask=None, prev_reward=None, return_state=False):
        # be sure obs is normalized if obs normalization is required
        shared_state = shared_state or self._env_output.obs['shared_state']
        mask = 1. - self._env_output.reset if mask is None else mask
        state = self._state.value_h if state is None else state
        shared_state, kwargs = self._add_memory_state_to_kwargs(
            shared_state, mask, state=state, prev_reward=prev_reward)
        kwargs['return_state'] = return_state
        out = self.model.compute_value(shared_state, **kwargs)
        return tf.nest.map_structure(lambda x: x.numpy(), out)

    @tf.function
    def _learn(self, obs, shared_state, action_mask, action, value, traj_ret, advantage, 
            logpi, state=None, life_mask=None, mask=None, prev_action=None, prev_reward=None):
        actor_state, value_state = state

        actor_terms = self._learn_actor(obs, action_mask, action, advantage, logpi, 
            actor_state, life_mask, mask, prev_action, prev_reward)
        value_terms = self._learn_value(shared_state, value, traj_ret, 
            value_state, life_mask, mask, prev_action, prev_reward)

        terms = {**actor_terms, **value_terms}

        return terms

    def _learn_actor(self, obs, action_mask, action, advantage, logpi, 
            actor_state=None, life_mask=None, mask=None, prev_action=None, prev_reward=None):
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
                self._policy_life_mask and life_mask)
            actor_loss = policy_loss - self._entropy_coef * entropy
        
        actor_norm = self._actor_opt(tape, actor_loss)
        terms = dict(
            actor_norm=actor_norm,
            entropy=entropy, 
            kl=kl, 
            p_clip_frac=p_clip_frac,
            ppo_loss=policy_loss,
            actor_loss=actor_loss,
        )

        return terms

    def _learn_value(self, shared_state, value, traj_ret, 
            value_state=None, life_mask=None, mask=None, prev_action=None, prev_reward=None):
        old_value = value
        with tf.GradientTape() as tape:
            x_value, _ = self.model.encode(
                shared_state, value_state, mask, 'value',
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

    def _process_obs(self, obs, update_rms=True, mask=None):
        for k in self._obs_names:
            v = obs[k]
            if update_rms:
                self.update_obs_rms(v, k, mask=mask)
            obs[k] = self.normalize_obs(v, k)   # mask is not necessary here as the action does not matter at all if the agent is dead
    
    def _divide_obs(self, obs):
        kwargs = {
            'shared_state': obs['shared_state'],
            'action_mask': obs['action_mask'].astype(np.bool)
        }
        obs = obs['obs']
        return obs, kwargs

    def _store_buffer_stats(self):
        self.store(**self.dataset.compute_mean_max_std('reward'))
        # self.store(**self.dataset.compute_mean_max_std('obs'))
        # self.store(**self.dataset.compute_mean_max_std('shared_state'))
        self.store(**self.dataset.compute_mean_max_std('advantage'))
        self.store(**self.dataset.compute_mean_max_std('value'))
        self.store(**self.dataset.compute_mean_max_std('traj_ret'))
        self.store(**self.dataset.compute_fraction('life_mask'))
        self.store(**self.dataset.compute_fraction('action_mask'))
        self.store(**self.dataset.compute_fraction('mask'))
        self.store(**self.dataset.compute_fraction('discount'))
