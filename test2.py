import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from utility.tf_utils import reduce_mean
from utility.rl_loss import kl_from_distributions


x = tf.Variable(np.log([[.3, .7]]), trainable=True)
y = tf.Variable(np.log([[.5, .5]]))
px = tf.nn.softmax(x)
py = tf.nn.softmax(y)
print('px', px.numpy(), 'py', py.numpy())
raw_kl = tf.math.log(px / py) * px
kl = tf.reduce_sum(raw_kl)
gradx = raw_kl - px * kl
raw_ratio = px / py * py
ratio = tf.reduce_sum(raw_ratio)
grady = py * ratio - raw_ratio
print('gradx', gradx.numpy())
print('grady', grady.numpy())
with tf.GradientTape(persistent=True) as tape:
    dx = tfd.Categorical(x)
    dy = tfd.Categorical(y)
    loss = dx.kl_divergence(dy)
    print('loss', loss)
    loss = tf.reduce_mean(loss)
    loss2 = kl_from_distributions(pi1=tf.nn.softmax(x), pi2=tf.nn.softmax(y))
print('grad from tfp.KL', *tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss, [x, y])), sep='\n')
print('grad from manual KL', *tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss2, [x, y])), sep='\n')
