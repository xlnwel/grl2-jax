import collections


# LSTM states
LSTMState = collections.namedtuple('LSTMState', ['h', 'c'])
# GRU states
GRUState = collections.namedtuple('GRUState', ['h'])
