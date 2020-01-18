from time import time
import numpy as np
import ray


@ray.remote(num_cpus=1)
def f(x, y):
    start = time()
    while True:
        x += y
        if np.mean(x) > 1000:
            break
    return time() - start

if __name__ == '__main__':
    # I intend to make x and y large to increase the cpu usage.
    x = np.random.rand(1000, 10000)
    y = np.random.uniform(0, 3, (1000, 10000))
    print('x mean:', np.mean(x))
    print('y mean:', np.mean(y))
    for n in range(1, 4):
        ray.init()

        start = time()
        result = ray.get([f.remote(x, y) for _ in range(n)])

        print('Num of workers:', n)
        # print('Run time:', result)
        print('Average run time:', np.mean(result))
        print('Ray run time:', time() - start)
        ray.shutdown()
