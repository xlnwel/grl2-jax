import os
import time
import tensorflow as tf
import ray

from distributed.coordinator import Coordinator
from distributed.remote.monitor import create_central_monitor
from utility.ray_setup import sigint_shutdown_ray


def main(config):
    gpus = tf.config.list_physical_devices('GPU')
    ray.init(num_cpus=os.cpu_count(), num_gpus=len(gpus))
    print('Ray available resources:', ray.available_resources())

    sigint_shutdown_ray()

    monitor = create_central_monitor(config)
    
    coordinator = Coordinator(config)
    coordinator.start(monitor)

    while not ray.get(monitor.is_over.remote()):
        time.sleep(60)
        monitor.record.remote()

    ray.shutdown()
