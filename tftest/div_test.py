import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utility import rl_loss

class MetaGradientTest(tf.test.TestCase):
    def setUp(self):
        return super().setUp()

    def testKLCategorical(self):
        # First
        x = tf.random.uniform((2, 10), 0, 3)
        y = tf.random.uniform((2, 10), 0, 3)
        x = x / tf.reduce_sum(x, -1, keepdims=True)
        y = y / tf.reduce_sum(y, -1, keepdims=True)
        tf.debugging.assert_equal(tf.reduce_sum(x, -1), 1.)
        tf.debugging.assert_equal(tf.reduce_sum(y, -1), 1.)
        x = tf.Variable(tf.math.log(x), trainable=True)
        y = tf.Variable(tf.math.log(y))
        px = tf.nn.softmax(x)
        py = tf.nn.softmax(y)
        raw_kl = tf.math.log(px / py) * px
        kl = tf.reduce_sum(raw_kl)
        gradx = raw_kl - px * kl
        raw_ratio = px / py * py
        ratio = tf.reduce_sum(raw_ratio)
        grady = py * ratio - raw_ratio
        with tf.GradientTape(persistent=True) as tape:
            dx = tfd.Categorical(x)
            dy = tfd.Categorical(y)
            loss1 = dx.kl_divergence(dy)
            loss1 = tf.reduce_mean(loss1)
            kl, _, loss2 = rl_loss.compute_kl(
                kl_type='forward', 
                kl_coef=1, 
                pi1=tf.nn.softmax(x), 
                pi2=tf.nn.softmax(y), 
            )
            tf.debugging.assert_near(loss1, loss2)
        gradx1, grady1 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss1, [x, y]))
        gradx2, grady2 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss2, [x, y]))
        tf.debugging.assert_near(gradx, gradx1)
        tf.debugging.assert_near(gradx, gradx2)
        tf.debugging.assert_near(grady, grady1)
        tf.debugging.assert_near(grady, grady2)

    def testJSCategorical(self):
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
        x = tf.Variable(tf.math.log([.15, .1, .3, .45]), dtype=tf.float32)
        y = tf.Variable(tf.math.log([.4, .3, .2, .1]), dtype=tf.float32)
        px = tf.nn.softmax(x)
        py = tf.nn.softmax(y)
        raw_ratiox = tf.math.log(2 * px / (px + py)) * px
        raw_ratioy = tf.math.log(2 * py / (px + py)) * py
        gradx = .5 * (raw_ratiox - px * tf.reduce_sum(raw_ratiox, -1, keepdims=True))
        grady = .5 * (raw_ratioy - py * tf.reduce_sum(raw_ratioy, -1, keepdims=True))
        with tf.GradientTape(persistent=True) as tape:
            px = tf.nn.softmax(x)
            py = tf.nn.softmax(y)
            dx = tfd.Categorical(probs=px)
            dy = tfd.Categorical(probs=py)
            avg = tfd.Categorical(probs=(px+py) / 2)
            loss1 = dx.kl_divergence(avg) + dy.kl_divergence(avg)
            loss1 = .5 * loss1
            loss1 = tf.reduce_mean(loss1)
            _, _, loss2 = rl_loss.compute_js(
                js_type='exact', 
                js_coef=1, 
                pi1=px, 
                pi2=py
            )
            tf.debugging.assert_near(loss1, loss2)
            loss3 = js_from_sample(logp=x, logq=y)
        gradx1, grady1 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss1, [x, y]))
        gradx2, grady2 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss2, [x, y]))
        gradx3 = tf.nest.map_structure(lambda x: x.numpy(), tape.gradient(loss3, x))
        tf.debugging.assert_near(gradx, gradx1)
        tf.debugging.assert_near(gradx, gradx2)
        tf.debugging.assert_near(gradx, gradx3)
        tf.debugging.assert_near(grady, grady1)
        tf.debugging.assert_near(grady, grady2)
