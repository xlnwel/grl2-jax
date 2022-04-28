import ray
import tensorflow as tf

from core.tf_config import configure_gpu


layers = tf.keras.layers.Dense(10)
opt = tf.optimizers.Adam(1e-4)
@tf.function
def optimize(x):
    with tf.GradientTape() as tape:
        y = layers(x)
        loss = tf.reduce_mean((1 - y)**2)
    
    grads = tape.gradient(loss, layers.variables)
    opt.apply_gradients(zip(grads, layers.variables))


if __name__ == '__main__':
    configure_gpu(0)
    while True:
        x = tf.random.normal((100, 100))
        optimize(x)
