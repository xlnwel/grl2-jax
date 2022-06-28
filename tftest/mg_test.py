import tensorflow as tf
from optimizers.adam import Adam

from optimizers.rmsprop import RMSprop


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

    def testCustomOptimizers(self):
        for cls in [RMSprop, Adam]:
            inner_opt = cls(1)
            meta_tape = tf.GradientTape(persistent=True)
            inner_tape = tf.GradientTape(persistent=True)

            l = tf.keras.layers.Dense(1)
            c = tf.Variable(1,dtype=tf.float32)
            def inner(x, tape):
                with tape:
                    x = l(x)
                    loss = tf.reduce_mean(c * x**2)
                grads = tape.gradient(loss, l.variables)
                
                return grads, loss

            def compute_hessian_vector(tape, vars1, vars2, grads):
                with tape:
                    out = tf.reduce_sum([tf.reduce_sum(o * g) for o, g in zip(grads, vars2)])
                new_grads = tape.gradient(out, vars1)
                return new_grads
            
            def compute_grads_through_optmizer(tape, vars, grads, trans_grads, out_grads):
                out_grads = compute_hessian_vector(
                    tape, 
                    grads, 
                    trans_grads, 
                    out_grads
                )
                grads = compute_hessian_vector(
                    tape, 
                    vars, 
                    grads, 
                    out_grads
                )
                return grads
            @tf.function
            def outer(x):
                with meta_tape:
                    grads, loss = inner(x, inner_tape)
                    inner_opt.apply_gradients(zip(grads, l.variables))
                    x = l(x)
                    loss = tf.reduce_mean((1 - x)**2)
                trans_grads = inner_opt.get_transformed_grads()
                out_grads = meta_tape.gradient(loss, l.variables)
                out_grads1 = meta_tape.gradient(inner_opt.get_transformed_grads(), grads, 
                    output_gradients=out_grads)
                out_grads1 = meta_tape.gradient(grads, c, output_gradients=out_grads1)
                out_grads2 = compute_grads_through_optmizer(
                    meta_tape, c, grads, trans_grads, out_grads)
                tf.debugging.assert_equal(out_grads1, out_grads2)

            x = tf.random.uniform((2, 3))
            outer(x)

    def testTwoStepCustomOptimizers(self):
        for cls in [RMSprop, Adam]:
            inner_opt = cls(1)
            meta_tape = tf.GradientTape(persistent=True)
            inner_tape = tf.GradientTape(persistent=True)

            l = tf.keras.layers.Dense(1)
            c = tf.Variable(1,dtype=tf.float32)

            def inner(x, tape):
                with tape:
                    x = l(x)
                    loss = tf.reduce_mean(c * x**2)
                grads = tape.gradient(loss, l.variables)
                
                return grads

            def compute_hessian_vector(tape, vars1, vars2, grads):
                """ Compute the gradient of vars1
                vars2 must be a differentiable function of vars1
                """
                # print(vars2, grads)
                with tape:
                    out = tf.reduce_sum([tf.reduce_sum(v * g) for v, g in zip(vars2, grads)])
                new_grads = tape.gradient(out, vars1)
                return new_grads

            def compute_grads_through_optmizer(tape, vars, grads, trans_grads, out_grads):
                print(trans_grads, out_grads)
                out_grads = compute_hessian_vector(
                    tape, 
                    grads, 
                    trans_grads, 
                    out_grads
                )
                grads = compute_hessian_vector(
                    tape, 
                    vars, 
                    grads, 
                    out_grads
                )
                return grads

            @tf.function
            def outer(x):
                with meta_tape:
                    grads1 = inner(x, inner_tape)
                    inner_opt.apply_gradients(zip(grads1, l.variables))
                trans_grads1 = inner_opt.get_transformed_grads()
                vars1 = l.variables
                with meta_tape:
                    grads2 = inner(x, inner_tape)
                    inner_opt.apply_gradients(zip(grads2, l.variables))
                    x = l(x)
                    loss = tf.reduce_mean((1 - x)**2)
                trans_grads2 = inner_opt.get_transformed_grads()
                out_grads = meta_tape.gradient(loss, l.variables)
                grads_list = []
                grads_list.append(
                    compute_grads_through_optmizer(
                        meta_tape, c, grads2, trans_grads2, out_grads)
                )
                out_grads1 = meta_tape.gradient(
                    trans_grads2, vars1, 
                    output_gradients=out_grads)
                out_grads1 = meta_tape.gradient(
                    trans_grads1, grads1, 
                    output_gradients=out_grads1)
                out_grads1 = meta_tape.gradient(
                    grads1, c, output_gradients=out_grads1)
                out_grads2 = compute_hessian_vector(
                    meta_tape,
                    vars1, 
                    trans_grads2, 
                    out_grads
                )
                out_grads2 = compute_grads_through_optmizer(
                    meta_tape, c, grads1, trans_grads1, out_grads2)
                tf.debugging.assert_equal(out_grads1, out_grads2)

                return out_grads2

            x = tf.random.uniform((2, 3))
            outer(x)