from multiprocessing import Process


def run_process(func, *args, join=False, **kwargs):
    p = Process(target=func, args=args, kwargs=kwargs, daemon=True)
    p.start()
    if join:
        p.join()

    return p
