import tensorflow as tf

class MetaGradientTest(tf.test.TestCase):
    def setUp(self):
        return super().setUp()

    def testMetaGradients3Times(self):
        # First
        c = tf.Variable(1.)
        x = tf.convert_to_tensor([[.5, 1.]])
        l = tf.keras.layers.Dense(1, kernel_initializer='ones')
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(c)
            with tf.GradientTape() as t2:
                y = l(x)
                self.assertAllEqual(y, [[1.5]])
                loss = tf.reduce_mean(c * (1 - y)**2)
                self.assertAllEqual(loss, .25)
            g = t2.gradient(loss, l.variables)
            self.assertAllEqual(g[0], [[.5], [1.]])
            self.assertAllEqual(g[1], [1.])
            [v.assign(v - g) for v, g in zip(l.variables, g)]
            self.assertAllEqual(l.variables[0], [[.5], [0]])
            self.assertAllEqual(l.variables[1], [-1.])
            y = l(x)
            self.assertAllEqual(y, [[-.75]])
            loss = tf.reduce_mean((1 - y)**2)
            self.assertAllEqual(loss, 3.0625)
        g2 = t1.gradient(loss, l.variables)
        self.assertAllEqual(g2[0], [[-1.75], [-3.5]])
        self.assertAllEqual(g2[1], [-3.5])
        gc = t1.gradient(g, c)
        self.assertAllEqual(gc, 2.5)
        mgc = t1.gradient(g, c, output_gradients=g2)
        self.assertAllEqual(mgc, -7.875)

        # Second
        x = tf.convert_to_tensor([[1, .5]])
        x2 = tf.convert_to_tensor([[.5, .5]])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(c)
            with tf.GradientTape() as t2:
                y = l(x)
                loss = tf.reduce_mean(c * (1 - y)**2)
                self.assertAllEqual(y, [[-.5]])
                self.assertAllEqual(loss, 2.25)
            g = t2.gradient(loss, l.variables)
            self.assertAllEqual(g[0], [[-3], [-1.5]])
            self.assertAllEqual(g[1], [-3])
            [v.assign(v - g) for v, g in zip(l.variables, g)]
            self.assertAllEqual(l.variables[0], [[3.5], [1.5]])
            self.assertAllEqual(l.variables[1], [2])
            y = l(x2)
            self.assertAllEqual(y, [[4.5]])
            loss = tf.reduce_mean((1 - y)**2)
            self.assertAllEqual(loss, 12.25)
        g2 = t1.gradient(loss, l.variables)
        self.assertAllEqual(g2[0], [[3.5], [3.5]])
        self.assertAllEqual(g2[1], [7])
        mgc = t1.gradient(g, c, output_gradients=g2)
        self.assertAllEqual(mgc, -36.75)

        # Third
        c.assign_sub(mgc)
        x = tf.convert_to_tensor([[-1, 1]])
        x2 = tf.convert_to_tensor([[1., 2.]])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(c)
            with tf.GradientTape() as t2:
                y = l(x)
                loss = tf.reduce_mean(c * (1 - y)**2)
                self.assertAllEqual(y, [[0]])
                self.assertAllEqual(loss, 37.75)
            g = t2.gradient(loss, l.variables)
            self.assertAllEqual(g[0], [[75.5], [-75.5]])
            self.assertAllEqual(g[1], [-75.5])
            [v.assign(v - g) for v, g in zip(l.variables, g)]
            self.assertAllEqual(l.variables[0], [[-72], [77]])
            self.assertAllEqual(l.variables[1], [77.5])
            y = l(x2)
            self.assertAllEqual(y, [[159.5]])
            loss = tf.reduce_mean((1 - y)**2)
            self.assertAllEqual(loss, 25122.25)
        g2 = t1.gradient(loss, l.variables)
        self.assertAllEqual(g2[0], [[317], [634]])
        self.assertAllEqual(g2[1], [317])
        mgc = t1.gradient(g, c, output_gradients=g2)
        self.assertAllEqual(mgc, -1268)