import os
import tensorflow as tf

from core.elements.model import Model
from utility.file import source_file

# register ppo-related networks 
source_file(os.path.realpath(__file__).replace('model.py', 'nn.py'))


class PPOModel(Model):
    @tf.function
    def action(self, obs, evaluation=False, **kwargs):
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


def create_model(config, env_stats, name='ppo'):
    config['policy']['action_dim'] = env_stats.action_dim
    config['policy']['is_action_discrete'] = env_stats.is_action_discrete

    return PPOModel(config=config, name=name)
