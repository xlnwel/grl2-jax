import os, sys, signal
import psutil
import ray

from core.log import do_logging


def sigint_shutdown_ray():
    """ Shutdown ray when the process is terminated by ctrl+C """
    def handler(sig, frame):
        if ray.is_initialized():
            ray.shutdown()
            do_logging('ray has been shutdown by sigint', color='cyan')
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

def cpu_affinity(name=None):
    resources = ray.worker.get_resource_ids()
    if 'CPU' in resources:
        cpus = [v[0] for v in resources['CPU']]
        psutil.Process().cpu_affinity(cpus)
    else:
        cpus = []
        # raise ValueError(f'No cpu is available')
    if name:
        do_logging(f'CPUs corresponding to {name}: {cpus}', color='cyan')

def gpu_affinity(name=None):
    gpus = ray.get_gpu_ids()

    if gpus:
        if isinstance(gpus[0], str):
            gpus = [c for c in gpus[0] if c.isdigit()]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    if name:
        do_logging(f'GPUs corresponding to {name}: {gpus}', color='cyan')

def get_num_cpus():
    return len(ray.worker.get_resource_ids()['CPU'])
