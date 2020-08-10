"""
This file defines general CNN architectures used in RL
"""
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.mixed_precision.experimental import global_policy

from .cnns.utils import *


load_cnn(os.path.dirname(os.path.realpath(__file__)))
