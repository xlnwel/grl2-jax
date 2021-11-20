import os
import tensorflow as tf

from core.elements.model import Model
from core.tf_config import build
from utility.file import source_file

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOModel(Model):
    def _build(self, env_stats):
        TensorSpecs = dict(
            obs=((None, *env_stats['obs_shape']), env_stats['obs_dtype'], 'obs'),
            evaluation=False,
            return_eval_stats=False,
        )
        self.action = build(self.action, TensorSpecs)

    @tf.function
    def action(self, obs, evaluation=False, return_eval_stats=False):
        x, state = self.encode(obs)
        act_dist = self.policy(x, evaluation=evaluation)
        action = self.policy.action(act_dist, evaluation)

        if evaluation:
            return action, {}, state
        else:
            logpi = act_dist.log_prob(action)
            value = self.value(x)
            terms = {'logpi': logpi, 'value': value}

            return action, terms, state    # keep the batch dimension for later use

    @tf.function
    def compute_value(self, obs, state=None, mask=None):
        x, state = self.encode(obs, state, mask)
        value = self.value(x)
        return value, state

    def encode(self, x, state=None, mask=None):
        x = self.encoder(x)
        if hasattr(self, 'rnn'):
            x, state = self.rnn(x, state, mask)
            return x, state
        else:
            return x, None


def create_model(config, env_stats, name='ppo', to_build=False):
    config['policy']['action_dim'] = env_stats['action_dim']
    config['policy']['is_action_discrete'] = env_stats['is_action_discrete']

    return PPOModel(
        config=config, 
        env_stats=env_stats, 
        name=name, 
        to_build=to_build)
