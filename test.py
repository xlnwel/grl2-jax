import time
import tensorflow as tf
from tensorflow.keras import layers
import ray

from core.module import Module
from core.tf_config import configure_gpu
from utility.timer import timeit

class Net(Module):
    def __init__(self) -> None:
        super().__init__('name')
        self._layers = [
            layers.Dense(100), 
            layers.Dense(500), 
            layers.Dense(100), 
            layers.Dense(1)]


@ray.remote(num_gpus=.1, num_cpus=1)
class Actor:
    def __init__(self) -> None:
        self.net = None
    
    def set_weights(self, weights):
        self.net.set_weights(weights)
    
    def set_net(self, net):
        self.net = net
    
    def remove_net(self):
        self.net = None
    
    def call(self, x):
        return self.net(x)


if __name__ == '__main__':

    configure_gpu()
    ray.init()
    actor = Actor.remote()
    x = tf.random.normal((1, 3))
    net = Net()
    print(net(x))
    start = time.time()
    for _ in range(10):
        ray.get(actor.set_net.remote(net))
        print(ray.get(actor.call.remote(x)))
    print(time.time() - start)

    start = time.time()
    for _ in range(10):
        ray.get(actor.set_weights.remote(net.get_weights()))
        print(ray.get(actor.call.remote(x)))
    print(time.time() - start)

    ray.get(actor.remove_net.remote())
    ray.shutdown()
