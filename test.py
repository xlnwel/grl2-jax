import tensorflow as tf
from tensorflow.keras import layers
import ray
from tensorflow.python.eager.context import num_gpus

from core.module import Module
from core.tf_config import configure_gpu


class Net(Module):
    def __init__(self) -> None:
        super().__init__('name')
        self._layers = [
            layers.Dense(10000), 
            layers.Dense(50000), 
            layers.Dense(10000), 
            layers.Dense(1)]


@ray.remote(num_gpus=1, num_cpus=1)
class Actor:
    def __init__(self) -> None:
        self.net = None
    
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
    for _ in range(10):
        x = tf.random.normal((200, 3))
        net = Net()
        net(x)
        ray.get(actor.set_net.remote(net))
        print(ray.get(actor.call.remote(x)))
    ray.get(actor.remove_net.remote())
    ray.shutdown()
    print('start sleeping')
    import time
    time.sleep(100)
