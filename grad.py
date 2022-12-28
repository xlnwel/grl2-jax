import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tools.tf_utils import reduce_mean
from tools.jax_div import js_from_distributions

def kl_from_distributions(
    *,
    pi1=None, 
    pi2=None,
    pi1_mean=None,  
    pi1_std=None,  
    pi2_mean=None,  
    pi2_std=None,  
    pi_mask=None,
    sample_mask=None, 
    n=None
):
    if pi1 is None:
        d1 = tfd.MultivariateNormalDiag(pi1_mean, pi1_std)
        d2 = tfd.MultivariateNormalDiag(pi2_mean, pi2_std)
        kl = d1.kl_divergence(d2)
    else:
        log_pi1 = tf.math.log(tf.clip_by_value(pi1, 1e-10, 1))
        log_pi2 = tf.math.log(tf.clip_by_value(pi2, 1e-10, 1))
        log_ratio = log_pi1 - log_pi2
        if pi_mask is not None:
            log_ratio = tf.where(pi_mask, log_ratio, 0)
        tf.debugging.assert_all_finite(log_ratio, 'Bad log_ratio')
        kl = tf.reduce_sum(pi1 * log_ratio, axis=-1)
        d1 = tfd.Categorical(log_pi1)
        d2 = tfd.Categorical(log_pi2)
        kl1 = d1.kl_divergence(d2)
        tf.debugging.assert_near(kl1, kl)
    kl = reduce_mean(kl, mask=sample_mask, n=n)

    return kl


def js_from_sample(
    *,
    logp,
    logq, 
    sample_prob=1, 
    mask=None,
    n=None
):
    p = tf.nn.softmax(logp)
    q = tf.nn.softmax(logq)
    approx_js = .5 * tf.reduce_sum(p * 
        tf.stop_gradient(
            (tf.math.log(2.) + logp - tf.math.log(p+q)) / sample_prob), 
            -1
    )
    return approx_js

x = tf.Variable(np.log([.15, .1, .3, .45]), dtype=tf.float32)
y = tf.Variable(np.log([.4, .3, .2, .1]), dtype=tf.float32)
px = tf.nn.softmax(x)
py = tf.nn.softmax(y)
print('px', px.numpy(), 'py', py.numpy())
raw_ratiox = tf.math.log(2 * px / (px + py)) * px
raw_ratioy = tf.math.log(2 * py / (px + py)) * py
gradx = .5 * (raw_ratiox - px * tf.reduce_sum(raw_ratiox, -1, keepdims=True))
grady = .5 * (raw_ratioy - py * tf.reduce_sum(raw_ratioy, -1, keepdims=True))
print('gradx', gradx.numpy())
print('grady', grady.numpy())
opt = tf.keras.optimizers.SGD(1)
with tf.GradientTape(persistent=True) as tape:
    px = tf.nn.softmax(x)
    py = tf.nn.softmax(y)
    dx = tfd.Categorical(probs=px)
    dy = tfd.Categorical(probs=py)
    avg = tfd.Categorical(probs=(px+py) / 2)
    loss1 = dx.kl_divergence(avg) + dy.kl_divergence(avg)
    loss1 = .5 * loss1
    print('loss1', loss1)
    loss1 = tf.reduce_mean(loss1)
    loss2 = js_from_distributions(p=px, q=py)
    print('loss2', loss2)
    loss3 = js_from_sample(logp=x, logq=y)
    print('loss3', loss3)
    loss3 = tf.reduce_mean(loss3)
    tf.debugging.assert_near(loss1, loss2)
gradx1, grady1 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss1, [x, y]))
gradx2, grady2 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss2, [x, y]))
gradx3 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss3, x))
tf.debugging.assert_near(gradx, gradx1)
tf.debugging.assert_near(gradx, gradx2)
tf.debugging.assert_near(gradx, gradx3)
tf.debugging.assert_near(grady, grady1)
tf.debugging.assert_near(grady, grady2)
