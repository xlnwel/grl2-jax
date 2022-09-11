from multiprocessing import Process


def run_process(func, *args, join=False, **kwargs):
    p = Process(target=func, args=args, kwargs=kwargs, daemon=True)
    p.start()
    if join:
        p.join()

    return p

def run_ray_process(func, *args, join=False, **kwargs):
    import ray
    remote_func = ray.remote(func)
    pid = remote_func.remote(*args, **kwargs)
    if join:
        return ray.get(pid)

    return pid
