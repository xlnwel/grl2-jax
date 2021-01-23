from abc import ABC, abstractmethod
import os
import cloudpickle
import logging

from utility.display import pwc
from utility.utils import Every
from utility.timer import Timer
from core.log import *
from core.checkpoint import *
from core.decorator import override, agent_config


logger = logging.getLogger(__name__)

class AgentImpl(ABC):
    """ Restore & save """
    def restore(self):
        """ Restore the latest parameter recorded by ckpt_manager

        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            ckpt: An instance of tf.train.Checkpoint
            ckpt_path: The directory in which to write checkpoints
            name: optional name for print
        """
        restore(self._ckpt_manager, self._ckpt, self._ckpt_path, self._model_name)
        self.env_step = self._env_step.numpy()
        self.train_step = self._train_step.numpy()

    def save(self, print_terminal_info=False):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
        """
        self._env_step.assign(self.env_step)
        self._train_step.assign(self.train_step)
        save(self._ckpt_manager, print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, prefix=None, print_terminal_info=True):
        log(self._logger, self._writer, self._model_name, prefix=prefix, 
            step=step, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
            
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def store(self, **kwargs):
        store(self._logger, **kwargs)

    def get_raw_item(self, key):
        return get_raw_item(self._logger, key)

    def get_item(self, key, mean=True, std=False, min=False, max=False):
        return get_item(self._logger, key, mean=mean, std=std, min=min, max=max)

    def get_raw_stats(self):
        return get_raw_stats(self._logger)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return get_stats(self._logger, mean=mean, std=std, min=min, max=max)
    
    def contains_stats(self, key):
        return contains_stats(self._logger, key)

    def print_construction_complete(self):
        pwc(f'{self.name.upper()} is constructed...', color='cyan')


class BaseAgent(AgentImpl):
    """ Initialization """
    @agent_config
    def __init__(self, *, env, dataset):
        super().__init__()
        self.dataset = dataset

        self._obs_shape = env.obs_shape
        self._action_shape = env.action_shape
        self._action_dim = env.action_dim

        # intervals between calling self._summary
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

        self._add_attributes(env, dataset)
        self._construct_optimizers()
        self._build_learn(env)
        self._sync_nets()
    
    def _add_attributes(self, env, dataset):
        self._sample_timer = Timer('sample')
        self._train_timer = Timer('train')

    @abstractmethod
    def _construct_optimizers(self):
        raise NotImplementedError

    @abstractmethod
    def _build_learn(self, env):
        raise NotImplementedError

    def _sync_nets(self):
        pass

    def _summary(self, data, terms):
        """ Add non-scalar summaries """
        pass 

    """ Call """
    def __call__(self, env_output=(), evaluation=False, return_eval_stats=False):
        """ Call the agent to interact with the environment
        Args:
            obs: Observation(s), we keep a separate observation to for legacy reasons
            evaluation bool: evaluation mode or not
            env_output tuple: (obs, reward, discount, reset)
        """
        obs = env_output.obs
        if obs.ndim % 2 != 0:
            obs = np.expand_dims(obs, 0)    # add batch dimension
        assert obs.ndim in (2, 4), obs.shape

        obs, kwargs = self._process_input(obs, evaluation, env_output)
        out = self._compute_action(
            obs, **kwargs, 
            evaluation=evaluation, 
            return_eval_stats=return_eval_stats)
        out = self._process_output(obs, kwargs, out, evaluation)

        return out

    def _process_input(self, obs, evaluation, env_output):
        """Do necessary pre-process and produce inputs to model
        Args:
            obs: Observations with added batch dimension
        Returns: 
            obs: Pre-processed observations
            kwargs, dict: kwargs necessary to model  
        """
        return obs, {}
    
    def _compute_action(self, obs, **kwargs):
        return self.model.action(obs, **kwargs)
        
    def _process_output(self, obs, kwargs, out, evaluation):
        """Post-process output
        Args:
            obs: Pre-processed observations
            kwargs, dict: kwargs necessary to model  
            out (action, terms): Model output
        Returns:
            out: results supposed to return by __call__
        """
        return tf.nest.map_structure(lambda x: x.numpy(), out)

    @abstractmethod
    def _learn(self):
        raise NotImplementedError


class RMSBaseAgent(BaseAgent):
    @override(BaseAgent)
    def _add_attributes(self, env, dataset):
        super()._add_attributes(env, dataset)

        from utility.utils import RunningMeanStd
        self._normalized_axis = getattr(self, '_normalized_axis', (0, 1))
        self._normalize_obs = getattr(self, '_normalize_obs', False)
        self._normalize_reward = getattr(self, '_normalize_reward', True)
        self._normalize_reward_with_reversed_return = \
            getattr(self, '_normalize_reward_with_reversed_return', True)
        
        axis = tuple(self._normalized_axis)
        self._obs_rms = self._normalize_obs and RunningMeanStd(axis)
        self._reward_rms = self._normalize_reward and RunningMeanStd(axis)
        if self._normalize_reward_with_reversed_return:
            self._reverse_return = 0
        else:
            self._reverse_return = -np.inf
        self._rms_path = f'{self._root_dir}/{self._model_name}/rms.pkl'

        logger.info(f'Observation normalization: {self._normalize_obs}')
        logger.info(f'Reward normalization: {self._normalize_reward}')
        logger.info(f'Reward normalization with reversed return: {self._normalize_reward_with_reversed_return}')

    @override(BaseAgent)
    def _process_input(self, obs, evaluation, env_output):
        obs = self.normalize_obs(obs)
        return obs, {}

    """ Functions for running mean and std """
    def get_running_stats(self):
        obs_rms = self._obs_rms.get_stats() if self._normalize_obs else ()
        rew_rms = self._reward_rms.get_stats() if self._normalize_reward else ()
        return obs_rms, rew_rms

    @property
    def is_obs_or_reward_normalized(self):
        return self._obs_rms or self._reward_rms

    def update_obs_rms(self, obs):
        if self._normalize_obs:
            if obs.dtype == np.uint8 and \
                    getattr(self, '_image_normalization_warned', False):
                logger.warning('Image observations are normalized. Make sure you intentionally do it.')
                self._image_normalization_warned = True
            self._obs_rms.update(obs)

    def update_reward_rms(self, reward, discount=None):
        if self._normalize_reward:
            assert len(reward.shape) == len(self._normalized_axis), (reward.shape, self._normalized_axis)
            if self._normalize_reward_with_reversed_return:
                """
                Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
                section 9.3 (which is based on our Baselines code, haha)
                Motivation is that we'd rather normalize the returns = sum of future rewards,
                but we haven't seen the future yet. So we assume that the time-reversed rewards
                have similar statistics to the rewards, and normalize the time-reversed rewards.

                Quoted from
                https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/reward_normalizer.py
                Yeah, you may not find the pseudocode. That's why I quote:-)
                """
                assert discount is not None, \
                    f"Normalizing rewards with reversed return requires environment's reset signals"
                assert reward.ndim == discount.ndim == len(self._reward_rms.axis), \
                    (reward.shape, discount.shape, self._reward_rms.axis)
                self._reverse_return, ret = backward_discounted_sum(
                    self._reverse_return, reward, discount, self._gamma)
                self._reward_rms.update(ret)
            else:
                self._reward_rms.update(reward)

    def normalize_obs(self, obs):
        return self._obs_rms.normalize(obs) if self._normalize_obs else obs

    def normalize_reward(self, reward):
        return self._reward_rms.normalize(reward, subtract_mean=False) \
            if self._normalize_reward else reward

    @override(BaseAgent)
    def restore(self):
        if os.path.exists(self._rms_path):
            with open(self._rms_path, 'rb') as f:
                self._obs_rms, self._reward_rms, self._reverse_return = cloudpickle.load(f)
                logger.info(f'rms stats are restored from {self._rms_path}')
        super().restore()

    @override(BaseAgent)
    def save(self, print_terminal_info=False):
        with open(self._rms_path, 'wb') as f:
            cloudpickle.dump((self._obs_rms, self._reward_rms, self._reverse_return), f)
        super().save(print_terminal_info=print_terminal_info)


def backward_discounted_sum(prev_ret, reward, discount, gamma):
    assert reward.ndim == discount.ndim, (reward.shape, discount.shape)
    if reward.ndim == 1:
        prev_ret = reward + gamma * prev_ret
        ret = prev_ret.copy()
        prev_ret *= discount
        return prev_ret, ret
    elif reward.ndim == 2:
        _nenv, nstep = reward.shape
        ret = np.zeros_like(reward)
        for t in range(nstep):
            ret[:, t] = prev_ret = reward[:, t] + gamma * prev_ret
            prev_ret *= discount[:, t]
        return prev_ret, ret
    else:
        raise ValueError(f'Unknown reward shape: {reward.shape}')
