import collections
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision.experimental import global_policy
from tensorflow_probability import distributions as tfd

from core.module import Module
from core.decorator import config
from nn.func import Encoder, LSTM, mlp
from nn.rnn import LSTMCell, LSTMState
from algo.dqn.nn import Q

LSTMState = collections.namedtuple('LSTMState', ['h', 'c'])



def create_model(config, env):
    action_dim = env.action_dim
    encoder_config = config['encoder']
    rnn_config = config['rnn']
    q_config = config['q']
    return dict(
        encoder=Encoder(encoder_config, name='encoder'),
        rnn=LSTM(rnn_config, name='rnn'),
        q=Q(q_config, action_dim, name='q'),
        target_encoder=Encoder(encoder_config, name='target_encoder'),
        target_rnn=LSTM(rnn_config, name='target_rnn'),
        target_q=Q(q_config, action_dim, name='target_q'),
    )
