import time
import asyncio

from utility.timer import timeit


async def sleep(t):
    await time.sleep(t)

class A:
    async def f(self, i):
        print(i)
        start = time.time()
        await sleep(1)
        print('f', time.time() - start)
        print(i)
        await sleep(1)
        print(i)
        return i

aa = [A() for _ in range(10)]
start = time.time()
async def main():
    return await asyncio.gather(*[a.f(i) for i, a in enumerate(aa)])
    
print(timeit(asyncio.run, main()))
print(time.time() - start)
